import torch
import torch.nn as nn
from torch.nn.functional import pad
from transformers import XLNetModel
import logging
from stack_lstm import StackLSTM
from pytorch_utils import make_embedding, make_ffnn


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()


class CorefModel(nn.Module):
    def __init__(self, config, device, class_weights=None, num_genres=None):
        super().__init__()
        self.config = config
        self.device = device

        # General parameters
        self.dropout = nn.Dropout(p=config['dropout_rate'])

        self.encoder = None
        self.encoder_mems = None
        if self.config['incremental_setting'] == 'part-inc':
            self.encoder = XLNetModel.from_pretrained(config['xlnet_pretrained_name_or_path'], use_mems_eval=False)
        elif self.config['incremental_setting'] == 'sent-inc':
            self.encoder = XLNetModel.from_pretrained(config['xlnet_pretrained_name_or_path'],
                                                      use_mems_train=True, mem_len=config['encoder_mem_len'])
        else:
            raise NotImplementedError()

        self.enc_emb_size = self.encoder.config.hidden_size
        self.mention_attn = make_ffnn(self.enc_emb_size, 0, output_size=1, dropout=self.dropout)

        # Mention Detector parameters
        self.left_boundary_stack = StackLSTM(self.enc_emb_size, config['stack_hidden_size'])
        self.action_hist_stack = StackLSTM(config['action_hist_hidden_size'], config['action_hist_hidden_size'])
        self.emb_action = make_embedding(4, embed_size=config['action_hist_hidden_size'])

        self.mention_parse_input_size = self.enc_emb_size + config['stack_hidden_size'] + config['action_hist_hidden_size']
        self.mention_parse_input_size += 2 * config['feature_emb_size']

        # Additional features
        self.num_genres = num_genres if num_genres else len(config['genres'])
        self.max_span_width = config['max_span_width']
        self.max_num_speakers = config['max_num_speakers']
        self.emb_span_width = make_embedding(self.max_span_width+1, config['feature_emb_size'])
        self.emb_genre = make_embedding(self.num_genres, config['feature_emb_size'])
        self.emb_speaker = make_embedding(self.max_num_speakers, config['feature_emb_size']) if config['use_speaker_embeds'] else None

        self.parse_action_ffnn = make_ffnn(self.mention_parse_input_size,
                                           [config['ffnn_size']] * config['ffnn_depth'],
                                           output_size=4, dropout=self.dropout)

        # Memory parameters
        self.mem, self.ent_counter, self.last_mention_idx = None, None, None
        self.new_cell_threshold = nn.Parameter(torch.tensor([self.config["new_cell_threshold"]]), requires_grad=False)
        if config['use_speaker_embeds']:
            self.memory_hidden_size = 3 * self.enc_emb_size + 2 * config['feature_emb_size']
        else:
            self.memory_hidden_size = 3 * self.enc_emb_size + config['feature_emb_size']
        self.emb_last_mem_action = make_embedding(4, config['feature_emb_size'])  # Coref, new entity, ignore, <s>
        self.emb_mention_distance = make_embedding(config['max_mention_distance'], config['feature_emb_size'])
        self.emb_entity_count = make_embedding(config['max_entity_count'], config['feature_emb_size'])

        mem_q_ffn_size = 3 * self.memory_hidden_size + 4 * config['feature_emb_size']
        self.memory_query_ffnn = make_ffnn(mem_q_ffn_size,
                                           [config['mem_ffnn_size']] * config['mem_ffnn_depth'],
                                           output_size=1,
                                           dropout=self.dropout)

        self.mention_cross_entropy_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.coref_cross_entropy_loss = nn.CrossEntropyLoss()

        self.predicted_clusters = []
        self.mention_action_history = []
        self.coref_action_history = []
        self.sent_start_idx, self.token_idx, self.action_idx, self.mention_idx = 0, 0, 0, 0

        self.update_steps = 0  # Internal use for debug
        self.debug = False

    def get_params(self, named=False):
        bert_based_param, task_param = [], []
        for name, param in self.named_parameters():
            # WARNING: the name of the encoder must match the string below or this will fail silently!
            if name.startswith('encoder'):
                to_add = (name, param) if named else param
                bert_based_param.append(to_add)
            else:
                to_add = (name, param) if named else param
                task_param.append(to_add)
        return bert_based_param, task_param

    def reset_state(self):
        """Re-initialize memory to empty."""
        self.mem = None
        self.ent_counter = None
        self.last_mention_idx = None

        self.left_boundary_stack.reset_state()
        self.action_hist_stack.reset_state()

        self.predicted_clusters = []
        self.mention_action_history = []
        self.coref_action_history = []
        self.sent_start_idx, self.token_idx, self.action_idx, self.mention_idx = 0, 0, 0, 0
        self.encoder_mems = None  # only used for XLNet

    def forward(self, *input):
        return self.get_predictions_and_loss(*input)

    def get_valid_action_mask(self, reached_end_token, current_action_state):
        """
        Set mask to determine valid actions at the current parser state.
        current_action_state:
            boolean tensor with dim. 4, indicating which actions have been called on the current token
            We use the mapping 0: ADV, 1: PUSH, 2: POP, 3: PEEK
        """
        valid_action_mask = torch.ones(4, device=self.device, dtype=torch.bool)
        if reached_end_token and self.left_boundary_stack:  # End token reached, stack non-empty
            valid_action_mask[0] = False
        if reached_end_token:  # Cannot call peek on final token
            valid_action_mask[3] = False
        if torch.any(current_action_state[1:]):  # Push cannot be called twice on same token, or directly after pop/peek
            valid_action_mask[1] = False
        if not self.left_boundary_stack:  # Cannot call pop/peek if stack empty
            valid_action_mask[2:] = False
        if current_action_state[3]:  # The only possibility after peek is advance
            valid_action_mask[1:] = False
        return valid_action_mask

    def get_mention_feature_embs(self, genre, token_idx, stack_position_id):
        genre_emb = self.emb_genre(genre)
        if stack_position_id:  # Stack is non-empty
            span_width = min(self.max_span_width - 1, token_idx - stack_position_id)
            span_width = torch.tensor(span_width, device=self.device, dtype=torch.long)
            width_emb = self.emb_span_width(span_width)
        else:  # Stack is empty
            width_emb = self.emb_span_width(torch.tensor(self.max_span_width, device=self.device, dtype=torch.long))
        return genre_emb, width_emb

    def get_mention_parse_prediction(self, buffer_emb, token_idx, sent_token_idx, genre, num_words, current_action_state):
        (stack_emb, _), stack_position_id = self.left_boundary_stack.top()
        (action_hist_emb, _), _ = self.action_hist_stack.top()

        genre_emb, width_emb = self.get_mention_feature_embs(genre, token_idx, stack_position_id)
        parser_state_emb = torch.cat([buffer_emb, stack_emb.squeeze(), action_hist_emb.squeeze(), genre_emb, width_emb])
        pred_action_logits = self.parse_action_ffnn(parser_state_emb)

        valid_action_mask = self.get_valid_action_mask(sent_token_idx >= num_words - 2, current_action_state)
        assert torch.any(valid_action_mask)  # Make sure there is a valid action possible, or cryptic errors will ensue

        valid_pred_action_logits = (torch.log(valid_action_mask.to(torch.float)) + pred_action_logits)
        top_score_action = torch.argmax(valid_pred_action_logits)
        return top_score_action, valid_pred_action_logits, valid_action_mask

    def update_parser_state(self, action, buffer_emb, token_idx, action_state):
        # Update parser state using chosen action
        left_boundary_rep, left_boundary_idx = None, None
        if action == 1:  # PUSH action
            action_state[1] = True
            self.left_boundary_stack.push(buffer_emb.unsqueeze(0), token_idx)
        elif action == 2:  # POP action
            action_state[2] = True
            (left_boundary_rep, _), left_boundary_idx = self.left_boundary_stack.pop()
        elif action == 3:  # PEEK action
            action_state[3] = True
            (left_boundary_rep, _), left_boundary_idx = self.left_boundary_stack.peek()
        else:  # ADV action
            action_state[:] = False
        self.action_hist_stack.push(self.emb_action(action).unsqueeze(0), token_idx)
        return left_boundary_rep, left_boundary_idx

    def construct_span_representation(self, span_reps, span_attn_scores_raw, span_width, speaker_id):
        span_attn_scores = nn.functional.softmax(span_attn_scores_raw, dim=0)
        head_attn_emb = torch.matmul(span_attn_scores, span_reps)

        span_width = torch.tensor(min(self.max_span_width - 1, span_width), device=self.device, dtype=torch.long)
        width_emb = self.emb_span_width(span_width)

        if self.config['use_speaker_embeds']:
            speaker_emb = self.emb_speaker(min(self.max_num_speakers - 1, speaker_id - 1))
            candidate_span_rep = torch.cat([span_reps[0], span_reps[-1], head_attn_emb, width_emb, speaker_emb])
        else:
            candidate_span_rep = torch.cat([span_reps[0], span_reps[-1], head_attn_emb, width_emb])
        return candidate_span_rep

    def get_coref_scores(self, query_vec, genre, curr_mention_idx, last_coref_action, memory_attn_context_vector=None):
        if self.mem is not None:
            num_cells = self.mem.shape[0]
            rep_query_vector = query_vec.repeat(num_cells, 1)  # num_cells x memory_hidden_size

            # Each feat has dim. num_cells x feature_emb_size
            genre_emb = self.emb_genre(genre).repeat(num_cells, 1)
            last_mention_decision = self.emb_last_mem_action(last_coref_action).repeat(num_cells, 1)
            mention_distance = self.emb_mention_distance(torch.clamp(curr_mention_idx - self.last_mention_idx,
                                                                     max=self.config['max_mention_distance'] - 1))
            num_ents = self.emb_entity_count(torch.clamp(self.ent_counter,
                                                         max=self.config['max_entity_count'] - 1))  # # of mentions in entity cluster
            feats = torch.cat([num_ents, mention_distance, last_mention_decision, genre_emb], dim=1)  # num_cells x 4 * feat

            if memory_attn_context_vector is None:
                rep_mem_context_vec = torch.tensor([], device=self.device)
            else:
                rep_mem_context_vec = memory_attn_context_vector.repeat(num_cells, 1)

            # num_cells x (3 or 4 * memory_hidden_size + 4 * feat)
            mem_query = torch.cat([rep_query_vector, self.mem, self.mem * rep_query_vector, rep_mem_context_vec, feats], dim=1)
            mem_scores = self.memory_query_ffnn(mem_query).view(1, -1)
            mem_scores_with_threshold = torch.cat([mem_scores, self.new_cell_threshold.unsqueeze(0)], dim=1)
            return mem_scores_with_threshold
        else:  # Initialize memory
            genre_emb = self.emb_genre(genre)
            last_mention_decision = self.emb_last_mem_action(last_coref_action)
            mention_distance = self.emb_mention_distance(torch.zeros(1, dtype=torch.long, device=self.device)).squeeze()
            num_ents = self.emb_entity_count(torch.ones(1, dtype=torch.long, device=self.device)).squeeze()
            feats = torch.cat([num_ents, mention_distance, last_mention_decision, genre_emb], dim=0)
            num_query_repeat = 3
            mem_query = torch.cat(num_query_repeat * [query_vec] + [feats], dim=0)
            mem_scores = self.memory_query_ffnn(mem_query).view(1, -1)
            return mem_scores

    def update_mem_cells(self, cell_idx, entity_vec, curr_mention_idx, mention_span, predicted_clusters):
        if self.mem is None:  # Initialize memory
            self.mem = entity_vec.unsqueeze(0)
            self.ent_counter = torch.ones(1, dtype=torch.long, device=self.device)
            self.last_mention_idx = torch.zeros(1, dtype=torch.long, device=self.device)
            predicted_clusters.append([mention_span])
            return torch.tensor(1, dtype=torch.long, device=self.device)

        num_mem_cells = self.mem.shape[0]
        if cell_idx < num_mem_cells:
            cell_mask = (torch.arange(0, self.mem.shape[0], device=self.device) == cell_idx).float()
            mask = torch.unsqueeze(cell_mask, dim=1)
            mask = mask.repeat(1, self.memory_hidden_size)
            avg_vec = (self.mem[cell_idx] * self.ent_counter[cell_idx] + entity_vec) / (self.ent_counter[cell_idx] + 1)
            self.mem = self.mem * (1 - mask) + mask * avg_vec
            self.ent_counter = self.ent_counter + cell_mask.long()
            self.last_mention_idx[cell_idx] = curr_mention_idx
            predicted_clusters[cell_idx].append(mention_span)
            return torch.tensor(0, dtype=torch.long, device=self.device)
        elif cell_idx == num_mem_cells:  # Create new cell
            self.mem = torch.cat([self.mem, entity_vec.unsqueeze(0)])
            self.ent_counter = torch.cat([self.ent_counter, torch.ones_like(self.ent_counter[0]).unsqueeze(0)])
            self.last_mention_idx = torch.cat([self.last_mention_idx,
                                               curr_mention_idx * torch.ones_like(self.last_mention_idx[0]).unsqueeze(0)])
            predicted_clusters.append([mention_span])
            return torch.tensor(1, dtype=torch.long, device=self.device)
        else:
            raise ValueError("Invalid clustering action. Num cells={}, Predicted cell idx={}. Expected value is "
                             "0 <= predicted cell idx <= num cells.".format(num_mem_cells, cell_idx))

    def get_predictions_and_loss(self, input_ids, input_mask, speaker_ids, sentence_lens, genre, sentence_map,
                                 gold_starts=None, gold_ends=None, gold_mention_cluster_map=None,
                                 gold_mention_parse_actions=None):
        if self.training:
            assert gold_mention_cluster_map is not None
            assert gold_starts is not None
            assert gold_ends is not None
            assert gold_mention_parse_actions is not None

        mention_loss, coref_loss, total_loss = 0.0, 0.0, 0.0
        speaker_ids = speaker_ids[input_mask.to(torch.bool)]
        doc_embed = self.encoder(input_ids, attention_mask=input_mask)['last_hidden_state']  # [num seg, num max tokens, emb size]
        input_mask = input_mask.to(torch.bool)
        doc_embed = doc_embed[input_mask]
        num_words = doc_embed.shape[0]

        mention_attn_scores_raw = self.mention_attn(doc_embed).squeeze()

        self.token_idx = 0
        last_coref_action = torch.tensor(2, dtype=torch.long, device=self.device)
        action_state = torch.zeros(4, dtype=torch.bool, device=self.device)
        while self.token_idx < num_words:
            buffer_emb = doc_embed[self.token_idx]
            mention_parse_input = (buffer_emb, self.token_idx, self.token_idx, genre, num_words, action_state)
            top_score_action, valid_pred_action_logits, valid_action_mask = self.get_mention_parse_prediction(*mention_parse_input)
            self.mention_action_history.append(top_score_action.item())

            if self.training:
                gold_parser_action = gold_mention_parse_actions[self.action_idx]
                chosen_action = gold_parser_action  # Teacher forcing
                action_loss = self.mention_cross_entropy_loss(valid_pred_action_logits.unsqueeze(0), gold_parser_action.unsqueeze(0))
                mention_loss += action_loss
            else:
                chosen_action = top_score_action  # Use parser prediction

            _, left_boundary_idx = self.update_parser_state(chosen_action, buffer_emb, self.token_idx, action_state)

            if chosen_action == 0:  # ADV action, advance token
                self.token_idx += 1

            if left_boundary_idx is not None:  # Model has identified a mention candidate
                mention_span = (left_boundary_idx, self.token_idx)
                span_rep_input = (doc_embed[left_boundary_idx: self.token_idx+1],
                                  mention_attn_scores_raw[left_boundary_idx: self.token_idx+1],
                                  self.token_idx - left_boundary_idx, speaker_ids[self.token_idx])
                candidate_span_rep = self.construct_span_representation(*span_rep_input)
                coref_scores = self.get_coref_scores(candidate_span_rep, genre, self.mention_idx, last_coref_action)
                predicted_coref_action = torch.argmax(coref_scores)

                if self.training:  # Teacher forcing or DAgger
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

        if self.training:
            mention_loss /= len(gold_mention_parse_actions)
            coref_loss /= len(self.coref_action_history) if len(self.coref_action_history) > 1 else 1.0
            total_loss = mention_loss + coref_loss
        predicted_clusters = [cluster for cluster in self.predicted_clusters if len(cluster) > 1]  # Discard singleton preds

        if self.debug:
            if self.update_steps % 20 == 0:
                logger.info('---------Debug step: {}---------'.format(self.update_steps))
                logger.info('Loss: {:.3f}'.format(total_loss))
        return self.mention_action_history, predicted_clusters, self.coref_action_history, total_loss

    def update_evaluator(self, predicted_clusters, gold_clusters, evaluator):
        predicted_clusters = [tuple(tuple(m) for m in cluster) for cluster in predicted_clusters]
        mention_to_predicted = {m: cluster for cluster in predicted_clusters for m in cluster}

        gold_clusters = [tuple(tuple(m) for m in cluster) for cluster in gold_clusters]
        mention_to_gold = {m: cluster for cluster in gold_clusters for m in cluster}
        evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
        return predicted_clusters
