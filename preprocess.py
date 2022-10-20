import argparse
import logging
import os
import re
import collections
import json
import conll
import util

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_word(word):
    if word == "/." or word == "/?":
        return word[1:]
    else:
        return word


def get_sentence_map(segments, sentence_end):
    sent_map = []
    sent_idx, subtok_idx = 0, 0
    for segment in segments:
        sent_map.append(sent_idx)  # [CLS]
        seg_len = len(segment) - 2
        for i in range(seg_len):
            sent_map.append(sent_idx)
            sent_idx += int(sentence_end[subtok_idx])
            subtok_idx += 1
        sent_map.append(sent_idx)  # [SEP]
    return sent_map


class DocumentState(object):
    def __init__(self, key, tokenizer_name, tokenizer):
        self.doc_key = key
        self.tokenizer_name = tokenizer_name
        self.cls_token = tokenizer.cls_token
        self.sep_token = tokenizer.sep_token
        self.tokens = []

        # Linear list mapped to subtokens without CLS, SEP
        self.subtokens = []
        self.subtoken_map = []
        self.token_end = []
        self.sentence_end = []
        self.info = []  # Only non-none for the first subtoken of each word

        # Linear list mapped to subtokens with CLS, SEP
        self.sentence_map = []

        # Segments (mapped to subtokens with CLS, SEP)
        self.segments = []
        self.segment_subtoken_map = []
        self.segment_info = []  # Only non-none for the first subtoken of each word
        self.speakers = []

        # Doc-level attributes
        self.pronouns = []
        self.clusters = collections.defaultdict(list)  # {cluster_id: [(first_subtok_idx, last_subtok_idx) for each mention]}
        self.coref_stacks = collections.defaultdict(list)

        # Mention parser actions
        self.mention_parser_actions = []
        self.mention_parse_stack = collections.defaultdict(list)

    def finalize(self):
        """ Extract clusters; fill other info e.g. speakers, pronouns """
        # Populate speakers from info
        subtoken_idx = 0
        for seg_info in self.segment_info:
            speakers = []
            for i, subtoken_info in enumerate(seg_info):
                if i == 0 or i == len(seg_info) - 1:
                    speakers.append('[SPL]')
                elif subtoken_info is not None:  # First subtoken of each word
                    speakers.append(subtoken_info[9])
                    # if subtoken_info[4] == 'PRP':  # Uncomment if needed
                    #     self.pronouns.append(subtoken_idx)
                else:
                    speakers.append(speakers[-1])
                subtoken_idx += 1
            self.speakers += [speakers]

        # Populate cluster
        first_subtoken_idx = 0  # Subtoken idx across segments
        subtokens_info = util.flatten(self.segment_info)
        seg_boundary_idx = [i for i, x in enumerate(util.flatten(self.segments)) if x == self.cls_token or x == self.sep_token]
        while first_subtoken_idx < len(subtokens_info):
            mention_parse_idx = first_subtoken_idx  # first_subtoken_idx <= mention_parse_idx <= last_subtoken_idx
            subtoken_info = subtokens_info[first_subtoken_idx]
            coref = subtoken_info[-2] if subtoken_info is not None else '-'
            last_subtoken_idx = first_subtoken_idx + subtoken_info[-1] - 1 if subtoken_info is not None else first_subtoken_idx
            if coref != '-':
                parts = coref.split('|')
                unary_pop_added = False
                for part_idx, part in enumerate(parts):
                    if part[0] == '(':
                        if part_idx == 0:
                            self.mention_parser_actions.append("PSH")
                            self.mention_parser_actions += ["ADV"] * (last_subtoken_idx - first_subtoken_idx)
                            mention_parse_idx = last_subtoken_idx

                        if part[-1] == ')':
                            cluster_id = int(part[1:-1])
                            self.clusters[cluster_id].append((first_subtoken_idx, last_subtoken_idx))

                            stack_pos = self.mention_parse_stack[first_subtoken_idx]
                            reduce_op = "POP" if not stack_pos else "PEK"

                            if reduce_op != "POP" or not unary_pop_added:  # Only allow one pop for unary mentions
                                self.mention_parser_actions.append(reduce_op)
                                unary_pop_added = True
                        else:
                            cluster_id = int(part[1:])
                            self.coref_stacks[cluster_id].append(first_subtoken_idx)
                            self.mention_parse_stack[first_subtoken_idx].append(cluster_id)
                    else:
                        cluster_id = int(part[:-1])
                        start = self.coref_stacks[cluster_id].pop()
                        self.clusters[cluster_id].append((start, last_subtoken_idx))

                        stack_pos = self.mention_parse_stack[start]
                        _ = stack_pos.pop()
                        reduce_op = "POP" if not stack_pos else "PEK"
                        if self.mention_parser_actions[-1] == "PEK" and reduce_op == "POP" and last_subtoken_idx == mention_parse_idx:
                            # Otherwise we get an error when mentions are not annotated in a nested manner
                            self.mention_parser_actions.insert(-1, reduce_op)
                        else:
                            self.mention_parser_actions += ["ADV"] * (last_subtoken_idx - mention_parse_idx) + [reduce_op]
                        mention_parse_idx = last_subtoken_idx
            else:
                self.mention_parser_actions += ["ADV"] * (last_subtoken_idx - first_subtoken_idx)
            if subtoken_info is not None or first_subtoken_idx in seg_boundary_idx:
                self.mention_parser_actions.append("ADV")
            first_subtoken_idx += 1

        # Merge clusters if any clusters have common mentions
        merged_clusters = []
        for cluster in self.clusters.values():
            existing = None
            for mention in cluster:
                for merged_cluster in merged_clusters:
                    if mention in merged_cluster:
                        existing = merged_cluster
                        break
                if existing is not None:
                    break
            if existing is not None:
                print("Merging clusters (shouldn't happen very often) in doc={}".format(self.doc_key))
                existing.update(cluster)
            else:
                merged_clusters.append(set(cluster))

        merged_clusters = [list(cluster) for cluster in merged_clusters]
        all_mentions = util.flatten(merged_clusters)
        sentence_map = get_sentence_map(self.segments, self.sentence_end)
        subtoken_map = util.flatten(self.segment_subtoken_map)

        # Sanity check
        assert len(all_mentions) == len(set(all_mentions))  # Each mention unique
        # Below should have length: # all subtokens with CLS, SEP in all segments
        num_all_seg_tokens = len(util.flatten(self.segments))
        assert num_all_seg_tokens == len(util.flatten(self.speakers))
        assert num_all_seg_tokens == len(subtoken_map)
        assert num_all_seg_tokens == len(sentence_map)

        mention_parse_check, mention_set, computed_mention_set = self.verify_mention_parse_actions(all_mentions)
        if not mention_parse_check:
            missing_mentions = mention_set.difference(computed_mention_set)
            erroneous_mentions = computed_mention_set.difference(mention_set)
            print("Doc key={}, missing mentions={}, erroneous mentions={}"
                  .format(self.doc_key, missing_mentions, erroneous_mentions))

        return {
            "doc_key": self.doc_key,
            "tokens": self.tokens,
            "sentences": self.segments,
            "speakers": self.speakers,
            "constituents": [],
            "ner": [],
            "clusters": merged_clusters,
            'sentence_map': sentence_map,
            "subtoken_map": subtoken_map,
            'pronouns': self.pronouns,
            "mention_parse_actions": self.mention_parser_actions
        }

    def verify_mention_parse_actions(self, mentions):
        """
        Sanity check to ensure that the mention parse actions fully recover the gold mentions
        :param mentions: List of gold mentions
        :return: True, if mention parse actions recover all gold mentions, False otherwise
        """
        mention_set = set([tuple(mention) for mention in mentions])
        computed_mentions = []
        stack = []
        tokens = util.flatten(self.segments)
        i, j = 0, 0
        while i < len(tokens) and j < len(self.mention_parser_actions):
            if self.mention_parser_actions[j] == "PSH":
                stack.append(i)
            elif self.mention_parser_actions[j] == "POP":
                try:
                    computed_mentions.append([stack.pop(), i])
                except IndexError as e:
                    raise IndexError("Found empty stack at action_idx={}, token_idx={}".format(j, i)) from e
            elif self.mention_parser_actions[j] == "PEK":
                try:
                    computed_mentions.append([stack[-1], i])
                except IndexError as e:
                    raise IndexError("Found empty stack at action_idx={}, token_idx={}".format(j, i)) from e
            elif self.mention_parser_actions[j] == "ADV":
                i += 1
            else:
                raise ValueError("Invalid action: {}".format(self.mention_parser_actions[j]))

            j += 1

        computed_mention_set = set([tuple(mention) for mention in computed_mentions])
        return mention_set == computed_mention_set, mention_set, computed_mention_set


def split_into_segments(document_state: DocumentState, max_seg_len, constraints1, constraints2, tokenizer):
    """ Split into segments.
        Add subtokens, subtoken_map, info for each segment; add CLS, SEP in the segment subtokens
        Input document_state: tokens, subtokens, token_end, sentence_end, utterance_end, subtoken_map, info
    """
    curr_idx = 0  # Index for subtokens
    prev_token_idx = 0
    while curr_idx < len(document_state.subtokens):
        # Try to split at a sentence end point
        end_idx = min(curr_idx + max_seg_len - 1 - 2, len(document_state.subtokens) - 1)  # Inclusive
        while end_idx >= curr_idx and not constraints1[end_idx]:
            end_idx -= 1
        if end_idx < curr_idx:
            logger.info(f'{document_state.doc_key}: no sentence end found; split at token end')
            # If no sentence end point, try to split at token end point
            end_idx = min(curr_idx + max_seg_len - 1 - 2, len(document_state.subtokens) - 1)
            while end_idx >= curr_idx and not constraints2[end_idx]:
                end_idx -= 1
            if end_idx < curr_idx:
                logger.error('Cannot split valid segment: no sentence end or token end')

        segment = [tokenizer.cls_token] + document_state.subtokens[curr_idx: end_idx + 1] + [tokenizer.sep_token]
        document_state.segments.append(segment)

        subtoken_map = document_state.subtoken_map[curr_idx: end_idx + 1]
        document_state.segment_subtoken_map.append([prev_token_idx] + subtoken_map + [subtoken_map[-1]])

        document_state.segment_info.append([None] + document_state.info[curr_idx: end_idx + 1] + [None])

        curr_idx = end_idx + 1
        prev_token_idx = subtoken_map[-1]


def split_into_sentences(document_state: DocumentState, tokenizer):
    curr_idx = 0
    end_idx = 0
    prev_token_idx = 0
    while curr_idx < len(document_state.subtokens):
        while not document_state.sentence_end[end_idx]:
            end_idx += 1

        sentence = [tokenizer.cls_token] + document_state.subtokens[curr_idx: end_idx + 1] + [tokenizer.sep_token]
        document_state.segments.append(sentence)

        subtoken_map = document_state.subtoken_map[curr_idx: end_idx + 1]
        document_state.segment_subtoken_map.append([prev_token_idx] + subtoken_map + [subtoken_map[-1]])

        document_state.segment_info.append([None] + document_state.info[curr_idx: end_idx + 1] + [None])

        curr_idx = end_idx + 1
        prev_token_idx = subtoken_map[-1]
        end_idx += 1


def get_document(doc_key, doc_lines, seg_len, split_sents, tokenizer, tokenizer_name):
    """ Process raw input to finalized documents """
    document_state = DocumentState(doc_key, tokenizer_name, tokenizer)
    word_idx = -1

    # Build up documents
    for line in doc_lines:
        row = line.split()  # Columns for each token
        if len(row) == 0:
            document_state.sentence_end[-1] = True
        else:
            assert len(row) >= 12
            word_idx += 1
            word = normalize_word(row[3])
            subtokens = tokenizer.tokenize(word)
            document_state.tokens.append(word)
            document_state.token_end += [False] * (len(subtokens) - 1) + [True]
            for idx, subtoken in enumerate(subtokens):
                document_state.subtokens.append(subtoken)
                info = None if idx != 0 else (row + [len(subtokens)])
                document_state.info.append(info)
                document_state.sentence_end.append(False)
                document_state.subtoken_map.append(word_idx)

    # Split documents
    constraints1 = document_state.sentence_end

    if split_sents:
        split_into_sentences(document_state, tokenizer)
    else:
        split_into_segments(document_state, seg_len, constraints1, document_state.token_end, tokenizer)
    document = document_state.finalize()
    return document


def minimize_partition(partition, extension, args, tokenizer):
    input_path = os.path.join(args.input_dir, f'{partition}.english.{extension}')
    seg_len = 'sents' if args.split_sents else str(args.seg_len)
    output_path = os.path.join(args.output_dir, f'{partition}.ontonotes.{seg_len}.{args.tokenizer_name}.jsonlines')
    doc_count = 0
    logger.info(f'Minimizing {input_path}...')

    # Read documents
    documents = []  # [(doc_key, lines)]
    with open(input_path, 'r') as input_file:
        for line in input_file.readlines():
            begin_document_match = re.match(conll.BEGIN_DOCUMENT_REGEX, line)
            if begin_document_match:
                doc_key = conll.get_doc_key(begin_document_match.group(1), begin_document_match.group(2))
                documents.append((doc_key, []))
            elif line.startswith('#end document'):
                continue
            else:
                documents[-1][1].append(line)

    # Write documents
    with open(output_path, 'w') as output_file:
        for doc_key, doc_lines in documents:
            document = get_document(doc_key, doc_lines, args.seg_len, args.split_sents, tokenizer, args.tokenizer_name)
            output_file.write(json.dumps(document))
            output_file.write('\n')
            doc_count += 1
    logger.info(f'Processed {doc_count} documents to {output_path}')


def minimize_language(args):
    tokenizer = util.get_tokenizer(args.tokenizer_name)

    minimize_partition('dev', 'v4_gold_conll', args, tokenizer)
    minimize_partition('test', 'v4_gold_conll', args, tokenizer)
    minimize_partition('train', 'v4_gold_conll', args, tokenizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_name', type=str, default='xlnet-base-cased',
                        help='Name or path of the tokenizer/vocabulary')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory that contains conll files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--seg_len', type=int, default=384,
                        help='Segment length: 128, 256, 384, 512')
    parser.add_argument('--split_sents', action='store_true',
                        help='Split input into sentences, with [CLS] and [SEP] tokens.')

    args = parser.parse_args()
    logger.info(args)
    os.makedirs(args.output_dir, exist_ok=True)

    minimize_language(args)
