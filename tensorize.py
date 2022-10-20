import util
import numpy as np
import os
from os.path import join
import json
import pickle
import logging
import torch
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)


class CorefDataProcessor:
    def __init__(self, config):
        self.config = config
        self.data_source = config['data_source']
        self.eval_only = config['eval_only']

        self.max_seg_len = config['max_segment_len']
        self.sent_split = config['sent_split']
        self.data_dir = config['data_dir']
        self.encoder_type = config['xlnet_pretrained_name_or_path']
        self.tokenizer_name = config['tokenizer_name']

        self.tokenizer = None  # Lazy loading
        self.tensor_samples, self.stored_info = None, None  # For dataset samples; lazy loading

    def get_tensor_examples_from_custom_input(self, samples):
        """ For interactive samples; no caching """
        tensorizer = Tensorizer(self.config, self.tokenizer)
        tensor_samples = [tensorizer.tensorize_example(sample, False) for sample in samples]
        tensor_samples = [(doc_key, self.convert_to_torch_tensor(*tensor)) for doc_key, tensor in tensor_samples]
        return tensor_samples, tensorizer.stored_info

    def get_max_sent_len(self, samples):
        max_len = 0
        for example in samples:
            max_sent_len = max([len(sent) for sent in example['sentences']])
            if max_sent_len > max_len:
                max_len = max_sent_len
        return max_len

    def get_tensor_examples(self):
        """ For dataset samples """
        cache_path = self.get_cache_path()
        if os.path.exists(cache_path):
            # Load cached tensors if exists
            with open(cache_path, 'rb') as f:
                self.tensor_samples, self.stored_info = pickle.load(f)
                logger.info('Loaded tensorized examples from cache')
        else:
            # Generate tensorized samples
            self.tokenizer = util.get_tokenizer(self.config['tokenizer_name'])
            dataset_names = ['dev', 'test'] if self.eval_only else ['train', 'dev', 'test']
            self.tensor_samples = {}
            if self.sent_split:
                paths = {
                    dataset_name: join(self.data_dir,
                                       f'{dataset_name}.{self.data_source}.sents.{self.tokenizer_name}.jsonlines')
                    for dataset_name in dataset_names
                }
            else:
                paths = {
                    dataset_name: join(self.data_dir,
                                       f'{dataset_name}.{self.data_source}.{self.max_seg_len}.{self.tokenizer_name}.jsonlines')
                    for dataset_name in dataset_names
                }

            tensorizer = Tensorizer(self.config, self.tokenizer)
            for split, path in paths.items():
                logger.info('Tensorizing examples from {}; results will be cached)'.format(path))
                is_training = (split == 'train')
                with open(path, 'r') as f:
                    samples = [json.loads(line) for line in f.readlines()]
                if (self.sent_split and is_training) or self.eval_only:  # train must be first and have the longest sentence
                    tensorizer.max_seg_len = self.get_max_sent_len(samples)
                tensor_samples = [tensorizer.tensorize_example(sample, is_training) for sample in samples]
                self.tensor_samples[split] = [(doc_key, self.convert_to_torch_tensor(*tensor)) for doc_key, tensor in tensor_samples]
            if not self.eval_only:
                class_weights = compute_class_weight(class_weight='balanced', classes=np.arange(4),
                                                     y=np.array(tensorizer.mention_parse_action_labels))
                tensorizer.stored_info['class_weights'] = torch.tensor(class_weights, dtype=torch.float)

            self.stored_info = tensorizer.stored_info
            # Cache tensorized samples
            with open(cache_path, 'wb') as f:
                pickle.dump((self.tensor_samples, self.stored_info), f)
        return self.tensor_samples.get('train', None), self.tensor_samples['dev'], self.tensor_samples['test']

    def get_stored_info(self):
        return self.stored_info

    @classmethod
    def convert_to_torch_tensor(cls, input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map,
                                gold_starts, gold_ends, gold_mention_cluster_map, gold_mention_parse_actions):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        speaker_ids = torch.tensor(speaker_ids, dtype=torch.long)
        sentence_len = torch.tensor(sentence_len, dtype=torch.long)
        genre = torch.tensor(genre, dtype=torch.long)
        sentence_map = torch.tensor(sentence_map, dtype=torch.long)
        gold_starts = torch.tensor(gold_starts, dtype=torch.long)
        gold_ends = torch.tensor(gold_ends, dtype=torch.long)
        gold_mention_cluster_map = torch.tensor(gold_mention_cluster_map, dtype=torch.long)
        gold_mention_parse_actions = torch.tensor(gold_mention_parse_actions, dtype=torch.long)
        return input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map, \
               gold_starts, gold_ends, gold_mention_cluster_map, gold_mention_parse_actions

    def get_cache_path(self):
        if self.sent_split:
            filename = f'cached.{self.data_source}.{self.encoder_type}.sents.bin'
        else:
            filename = f'cached.{self.data_source}.{self.encoder_type}.{self.max_seg_len}.bin'
        cache_path = join(self.data_dir, filename)
        return cache_path


class Tensorizer:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        self.mention_parse_action_map = {"ADV": 0, "PSH": 1, "POP": 2, "PEK": 3}

        # Will be used in evaluation
        self.stored_info = {'tokens': {}, 'subtoken_maps': {}, 'gold': {},
                            'genre_dict': {genre: idx for idx, genre in enumerate(config['genres'])}}

        self.mention_parse_action_labels = []
        self.max_seg_len = self.config['max_segment_len']

    def _tensorize_spans(self, spans):
        if len(spans) > 0:
            starts, ends = zip(*spans)
        else:
            starts, ends = [], []
        return np.array(starts), np.array(ends)

    def _tensorize_span_w_labels(self, spans, label_dict):
        if len(spans) > 0:
            starts, ends, labels = zip(*spans)
        else:
            starts, ends, labels = [], [], []
        return np.array(starts), np.array(ends), np.array([label_dict[label] for label in labels])

    def _get_speaker_dict(self, speakers):
        speaker_dict = {'UNK': 0, '[SPL]': 1}
        for speaker in speakers:
            if len(speaker_dict) > self.config['max_num_speakers']:
                pass  # 'break' to limit # speakers
            if speaker not in speaker_dict:
                speaker_dict[speaker] = len(speaker_dict)
        return speaker_dict

    def tensorize_example(self, example, is_training):
        # Mentions and clusters
        clusters = sorted(example['clusters'], key=util.sort_clusters)
        gold_mentions = sorted([tuple(mention) for mention in util.flatten(clusters)], key=lambda x: (x[1], -x[0]))
        gold_mention_map = {mention: idx for idx, mention in enumerate(gold_mentions)}
        gold_mention_cluster_map = np.zeros(len(gold_mentions))  # 0: no cluster
        for cluster_id, cluster in enumerate(clusters):
            for mention in cluster:
                gold_mention_cluster_map[gold_mention_map[tuple(mention)]] = cluster_id

        curr_highest_cluster = 0
        for cluster_mapping in gold_mention_cluster_map:
            if cluster_mapping > curr_highest_cluster:
                if cluster_mapping != curr_highest_cluster + 1:
                    raise AssertionError("Illegal mapping at doc: {}, at cluster {}".format(example['doc_key'], curr_highest_cluster))
                else:
                    curr_highest_cluster = cluster_mapping

        # Speakers
        speakers = example['speakers']
        speaker_dict = self._get_speaker_dict(util.flatten(speakers))

        # Sentences/segments
        sentences = example['sentences']  # Segments
        sentence_map = example['sentence_map']
        num_words = sum([len(s) for s in sentences])
        max_sentence_len = self.max_seg_len
        sentence_len = np.array([len(s) for s in sentences])

        # Bert input
        input_ids, input_mask, speaker_ids = [], [], []
        for idx, (sent_tokens, sent_speakers) in enumerate(zip(sentences, speakers)):
            sent_input_ids = self.tokenizer.convert_tokens_to_ids(sent_tokens)
            sent_input_mask = [1] * len(sent_input_ids)
            sent_speaker_ids = [speaker_dict[speaker] for speaker in sent_speakers]
            while len(sent_input_ids) < max_sentence_len:
                sent_input_ids.append(0)
                sent_input_mask.append(0)
                sent_speaker_ids.append(0)
            input_ids.append(sent_input_ids)
            input_mask.append(sent_input_mask)
            speaker_ids.append(sent_speaker_ids)
        input_ids = np.array(input_ids)
        input_mask = np.array(input_mask)
        speaker_ids = np.array(speaker_ids)
        assert num_words == np.sum(input_mask), (num_words, np.sum(input_mask))

        # Mention parse actions
        gold_mention_parse_action_strings = example.get('mention_parse_actions', ["ADV"] * num_words)
        gold_mentions_parse_actions = [self.mention_parse_action_map[gmpas] for gmpas in gold_mention_parse_action_strings]

        if is_training:
            self.mention_parse_action_labels += gold_mentions_parse_actions

        # Keep info to store
        doc_key = example['doc_key']
        self.stored_info['subtoken_maps'][doc_key] = example.get('subtoken_map', None)
        self.stored_info['gold'][doc_key] = example['clusters']

        # Construct example
        genre_str = "tc" if self.config['eval_only'] else doc_key[:2]  # Hardcoded for dialogue if eval only
        genre = self.stored_info['genre_dict'].get(genre_str, 0)
        gold_starts, gold_ends = self._tensorize_spans(gold_mentions)
        example_tensor = (input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map,
                          gold_starts, gold_ends, gold_mention_cluster_map, gold_mentions_parse_actions)

        return doc_key, example_tensor
