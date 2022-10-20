from os import makedirs
from os.path import join
import numpy as np
import pyhocon
import logging
import torch
import random
from transformers import BertTokenizer, XLNetTokenizer

logger = logging.getLogger(__name__)


def flatten(l):
    return [item for sublist in l for item in sublist]


def sort_clusters(cluster):
    """Make sure clusters are processed in order of shift-reduce,
    so that gold memory cells are created in the correct order during training."""
    sorted_mentions = sorted(cluster, key=lambda x: (x[1], -x[0]))
    return sorted_mentions[0][1], -sorted_mentions[0][0]


def get_tokenizer(tokenizer_name):
    if tokenizer_name == 'bert-base-cased':
        return BertTokenizer.from_pretrained(tokenizer_name)
    elif tokenizer_name == 'xlnet-base-cased':
        return XLNetTokenizer.from_pretrained(tokenizer_name)
    else:
        raise ValueError("Invalid tokenizer name: {}".format(tokenizer_name))


def initialize_config(config_name):
    logger.info("Running experiment: {}".format(config_name))

    config = pyhocon.ConfigFactory.parse_file("experiments.conf")[config_name]
    config['log_dir'] = join(config["log_root"], config_name)
    makedirs(config['log_dir'], exist_ok=True)

    config['tb_dir'] = join(config['log_root'], 'tensorboard')
    makedirs(config['tb_dir'], exist_ok=True)

    logger.info(pyhocon.HOCONConverter.convert(config, "hocon"))
    return config


def set_seed(seed, set_gpu=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if set_gpu and torch.cuda.is_available():
        # Necessary for reproducibility; lower performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)
    logger.info('Random seed is set to %d' % seed)
