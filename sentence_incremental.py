from model import CorefModel
import torch
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()


class SentenceIncremental(CorefModel):
    def __init__(self, config, device, class_weights=None, num_genres=None):
        super().__init__(config, device, class_weights, num_genres)

    def reshape_input_to_k_sents(self, k, input_ids, input_mask, speaker_ids, sentence_lens):
        num_sents, max_sent_len = input_ids.shape
        num_pad_sents = k - num_sents % k
        if num_pad_sents:
            padding = torch.zeros((num_pad_sents, max_sent_len), device=self.device, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, padding])
            input_mask = torch.cat([input_mask, padding])
            speaker_ids = torch.cat([speaker_ids, padding])
            sentence_lens = torch.cat([sentence_lens, torch.zeros(num_pad_sents, device=self.device, dtype=sentence_lens.dtype)])
        input_ids = input_ids.view(-1, k * max_sent_len)
        input_mask = input_mask.view(-1, k * max_sent_len)
        speaker_ids = speaker_ids.view(-1, k * max_sent_len)
        sentence_lens = sentence_lens.view(-1, k).sum(dim=1)
        return input_ids, input_mask, speaker_ids, sentence_lens

    def get_predictions_and_loss(self, input_ids, input_mask, speaker_ids, sentence_lens, genre, sentence_map,
                                 gold_starts=None, gold_ends=None, gold_mention_cluster_map=None,
                                 gold_mention_parse_actions=None):
        if self.training:
            assert gold_mention_cluster_map is not None
            assert gold_starts is not None
            assert gold_ends is not None
            assert gold_mention_parse_actions is not None

        if self.config['reshape_k'] > 0:
            reshape_k_input = (self.config['reshape_k'], input_ids, input_mask, speaker_ids, sentence_lens)
            input_ids, input_mask, speaker_ids, sentence_lens = self.reshape_input_to_k_sents(*reshape_k_input)

        # We must reset action hist stack at beginning of input to avoid gradient errors in multi-batch setting
        self.action_hist_stack.reset_state()
        if self.mem is not None:  # Memory cells must similarly be detached.
            self.mem = self.mem.detach()

        mem_lens = []
        mention_loss, coref_loss, total_loss = 0.0, 0.0, 0.0
        input_mask = input_mask.to(torch.bool)

        num_sents = input_ids.shape[0]
        num_words = input_ids[input_mask].shape[0]
        msgl = self.config['max_segment_len']
        for sent_idx in range(num_sents):
            sent_input_mask = input_mask[sent_idx]
            sent_input_ids = input_ids[sent_idx][sent_input_mask]
            sent_speaker_id = speaker_ids[sent_idx][sent_input_mask]
            sent_len = sentence_lens[sent_idx]
            sent_token_idx = 0

            if self.config['reshape_k'] > 0:
                mem_len = max(0, msgl - sent_input_ids.shape[0]) if num_words > msgl else num_words - sent_input_ids.shape[0]
                mem_lens.append(mem_len)

                # Provide cutoff for mem len
                self.encoder_mems = [
                    mems[-mem_len:] for mems in self.encoder_mems
                ] if mem_len and self.encoder_mems is not None else None

            sent_embed_output = self.encoder(sent_input_ids.unsqueeze(0), mems=self.encoder_mems)
            self.encoder_mems = sent_embed_output.mems
            sent_embed = sent_embed_output.last_hidden_state.squeeze()

            mention_attn_scores_raw = self.mention_attn(sent_embed).squeeze()

            last_coref_action = torch.tensor(2, dtype=torch.long, device=self.device)
            action_state = torch.zeros(4, dtype=torch.bool, device=self.device)
            while sent_token_idx < sent_len:
                buffer_emb = sent_embed[sent_token_idx]
                mention_parse_input = (buffer_emb, self.token_idx, sent_token_idx, genre, sent_len, action_state)
                top_score_action, valid_pred_action_logits, valid_action_mask = self.get_mention_parse_prediction(*mention_parse_input)

                if self.training:
                    gold_parser_action = gold_mention_parse_actions[self.action_idx]
                    chosen_action = gold_parser_action  # Teacher forcing
                    action_loss = self.mention_cross_entropy_loss(valid_pred_action_logits.unsqueeze(0), gold_parser_action.unsqueeze(0))
                    mention_loss += action_loss
                else:
                    chosen_action = top_score_action  # Use parser prediction
                self.mention_action_history.append(chosen_action.item())
                _, left_boundary_idx = self.update_parser_state(chosen_action, buffer_emb, self.token_idx, action_state)

                if chosen_action == 0:  # ADV action, advance token
                    sent_token_idx += 1
                    self.token_idx += 1

                if left_boundary_idx is not None:  # Model has identified a mention candidate
                    mention_span = (left_boundary_idx, self.token_idx)
                    sent_left_boundary_idx = left_boundary_idx - self.sent_start_idx  # Adjust doc idx to sent idx
                    span_rep_input = (sent_embed[sent_left_boundary_idx: sent_token_idx + 1],
                                      mention_attn_scores_raw[sent_left_boundary_idx: sent_token_idx + 1],
                                      self.token_idx - left_boundary_idx, sent_speaker_id[sent_token_idx])
                    candidate_span_rep = self.construct_span_representation(*span_rep_input)
                    coref_scores = self.get_coref_scores(candidate_span_rep, genre, self.mention_idx, last_coref_action)
                    predicted_coref_action = torch.argmax(coref_scores)

                    if self.training:  # Teacher forcing
                        gold_coref_action = gold_mention_cluster_map[self.mention_idx]
                        chosen_coref_action = gold_coref_action
                        cluster_loss = self.coref_cross_entropy_loss(coref_scores, gold_coref_action.unsqueeze(0))
                        coref_loss += cluster_loss
                    else:  # Use parser prediction
                        chosen_coref_action = predicted_coref_action

                    last_coref_action = self.update_mem_cells(chosen_coref_action, candidate_span_rep, self.mention_idx,
                                                              mention_span, self.predicted_clusters)
                    self.coref_action_history.append(chosen_coref_action.item())
                    self.mention_idx += 1

                self.action_idx += 1
            self.sent_start_idx += sent_len

        if self.training:
            mention_loss /= len(gold_mention_parse_actions)
            coref_loss /= len(self.coref_action_history) if len(self.coref_action_history) > 1 else 1.0
            total_loss = mention_loss + coref_loss
        predicted_clusters = [cluster for cluster in self.predicted_clusters if len(cluster) > 1]  # Discard singletons

        if self.debug:
            if self.update_steps % 20 == 0:
                logger.info('---------Debug step: {}---------'.format(self.update_steps))
                logger.info('Loss: {:.3f}'.format(total_loss))
        return self.mention_action_history, predicted_clusters, self.coref_action_history, total_loss
