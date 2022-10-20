import logging
import random
import time
from collections import defaultdict
from datetime import datetime
from os.path import join

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW

import conll
import util
from metrics import CorefEvaluator
from model import CorefModel
from sentence_incremental import SentenceIncremental
from tensorize import CorefDataProcessor

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()


class Runner:
    def __init__(self, config_name, gpu_id=None, seed=None):
        self.name = config_name
        self.name_suffix = datetime.now().strftime('%b%d_%H-%M-%S')
        self.gpu_id = gpu_id
        self.seed = seed

        # Set up config
        self.config = util.initialize_config(config_name)

        # Set up logger
        log_path = join(self.config['log_dir'], 'log_' + self.name_suffix + '.txt')
        logger.addHandler(logging.FileHandler(log_path, 'a'))
        logger.info('Log file path: %s' % log_path)

        # Set up seed
        if seed:
            util.set_seed(seed)

        # Set up device
        self.device = torch.device('cpu' if gpu_id is None else f'cuda:{gpu_id}')

        # Set up data
        self.data = CorefDataProcessor(self.config)

    def initialize_model(self, saved_suffix=None, class_weights=None):
        if self.config['incremental_setting'] == 'part-inc':
            model = CorefModel(self.config, self.device, class_weights=class_weights)
        elif self.config['incremental_setting'] == 'sent-inc':
            model = SentenceIncremental(self.config, self.device, class_weights=class_weights)
        else:
            raise NotImplementedError("Invalid encoding method: {}".format(self.config['encoding_method']))

        if saved_suffix:
            self.load_model_checkpoint(model, saved_suffix)
        return model

    def train(self, model=None, saved_suffix=None):
        conf = self.config
        logger.info(conf)
        epochs, grad_accum = conf['num_epochs'], conf['gradient_accumulation_steps']

        # Set up dataset
        examples_train, examples_dev, examples_test = self.data.get_tensor_examples()
        total_update_steps = len(examples_train) * epochs // grad_accum
        stored_info = self.data.get_stored_info()

        if not model:
            class_weights = stored_info['class_weights']
            model = self.initialize_model(saved_suffix, class_weights=class_weights)
            logging.info("Initialized model with class weights: {}".format(class_weights))
        model.to(self.device)

        # Set up tensorboard
        tb_path = join(conf['tb_dir'], self.name + '_' + self.name_suffix)
        tb_writer = SummaryWriter(tb_path, flush_secs=30)
        logger.info('Tensorboard summary path: %s' % tb_path)

        # Set up optimizer and scheduler
        optimizers = self.get_optimizer(model)
        schedulers = self.get_scheduler(optimizers, total_update_steps)

        # Get model parameters for grad clipping
        bert_param, task_param = model.get_params()

        # Start training
        logger.info('*******************Training*******************')
        logger.info('Num samples: {}'.format(len(examples_train)))
        logger.info('Num epochs: {}'.format(epochs))
        logger.info('Gradient accumulation steps: {}'.format(grad_accum))
        logger.info('Total update steps: {}'.format(total_update_steps))

        batch_size = conf['sent_batch_size']
        loss_during_report = 0.0  # Effective loss during logging step
        nans_during_report = 0
        loss_history = []  # Full history of effective loss; length equals total update steps
        max_f1 = 0
        start_time = time.time()
        model.zero_grad()
        for epo in range(epochs):
            random.shuffle(examples_train)  # Shuffle training set
            for doc_key, example in examples_train:
                # Forward pass
                model.train()
                model.reset_state()

                loss_during_batch = []
                num_sents = example[0].shape[0]
                num_batches = (num_sents // batch_size) + bool(num_sents % batch_size) if batch_size else 1
                for batch_i in range(num_batches):
                    if batch_size:
                        batcheable_input, example_non_batched = example[:4], list(example[4:])
                        example_batched = [ex[batch_i*batch_size: (batch_i+1)*batch_size] for ex in batcheable_input]
                        example_gpu = [d.to(self.device) for d in example_batched + example_non_batched]
                    else:
                        example_gpu = [d.to(self.device) for d in example]
                    _, _, _, loss = model(*example_gpu)
                    loss /= num_batches
                    loss_during_batch.append(loss.item())
                    loss.backward()

                if conf['max_grad_norm']:
                    torch.nn.utils.clip_grad_norm_(bert_param, conf['max_grad_norm'])
                    torch.nn.utils.clip_grad_norm_(task_param, conf['max_grad_norm'])

                for optimizer in optimizers:
                    optimizer.step()
                model.zero_grad()
                for scheduler in schedulers:
                    scheduler.step()

                # Compute effective loss
                effective_loss = np.sum(loss_during_batch).item()
                if np.isnan(effective_loss):
                    nans_during_report += 1
                else:
                    loss_during_report += effective_loss
                loss_history.append(effective_loss)
                model.update_steps += 1

                # Report
                if len(loss_history) % conf['report_frequency'] == 0:
                    # Show avg loss during last report interval
                    avg_loss = loss_during_report / conf['report_frequency']
                    loss_during_report = 0.0
                    end_time = time.time()
                    steps_per_sec = conf['report_frequency'] / (end_time - start_time)
                    logger.info('Step {}: avg loss {:.2f}; steps/sec {:.2f}; nans count {}'.format(
                                len(loss_history), avg_loss, steps_per_sec, nans_during_report))
                    start_time = end_time
                    nans_during_report = 0

                    tb_writer.add_scalar('Training_Loss', avg_loss, len(loss_history))
                    tb_writer.add_scalar('Learning_Rate_Bert', schedulers[0].get_last_lr()[0], len(loss_history))
                    tb_writer.add_scalar('Learning_Rate_Task', schedulers[1].get_last_lr()[-1], len(loss_history))
                    # Evaluate
                    if len(loss_history) > 0 and len(loss_history) % conf['eval_frequency'] == 0:
                        f1, _ = self.evaluate(model,
                                              examples_dev,
                                              stored_info,
                                              len(loss_history),
                                              official=False,
                                              conll_path=self.config['conll_eval_path'],
                                              tb_writer=tb_writer)
                        if f1 > max_f1:
                            max_f1 = f1
                            self.save_model_checkpoint(model, len(loss_history))
                        logger.info('Eval max f1: %.2f' % max_f1)
                        start_time = time.time()

        logger.info('**********Finished training**********')
        logger.info('Actual update steps: {}'.format(len(loss_history)))

        # Wrap up
        tb_writer.close()
        return loss_history

    def evaluate(self, model, tensor_examples, stored_info, step, official=False, conll_path=None, tb_writer=None):
        logger.info('Step {}: evaluating on {} samples...'.format(step, len(tensor_examples)))
        model.to(self.device)
        evaluator = CorefEvaluator()
        doc_to_prediction = {}
        mention_action_counts = defaultdict(int)
        total_action_len = 0
        batch_size = self.config['sent_batch_size']

        model.eval()
        model.debug = False
        for i, (doc_key, example) in enumerate(tensor_examples):
            with torch.no_grad():
                model.reset_state()
                gold_clusters = stored_info['gold'][doc_key]
                example = example[:7]  # Strip out gold
                predicted_clusters = []
                mention_action_history_batch = []
                num_sents = example[0].shape[0]
                num_batches = (num_sents // batch_size) + bool(num_sents % batch_size) if batch_size else 1
                for batch_i in range(num_batches):
                    if batch_size:
                        batcheable_input, example_non_batched = example[:4], list(example[4:])
                        example_batched = [ex[batch_i * batch_size: (batch_i + 1) * batch_size] for ex in batcheable_input]
                        example_gpu = [d.to(self.device) for d in example_batched + example_non_batched]
                    else:
                        example_gpu = [d.to(self.device) for d in example]
                    predicted_mention_actions, predicted_clusters, predicted_coref_actions, _ = model(*example_gpu)
                    mention_action_history_batch += predicted_mention_actions

                mention_action_values, mention_doc_action_counts = np.unique(mention_action_history_batch, return_counts=True)
                for action_val, action_count in zip(mention_action_values, mention_doc_action_counts):
                    mention_action_counts[action_val] += action_count
                total_action_len += len(mention_action_history_batch)
                predicted_clusters = model.update_evaluator(predicted_clusters, gold_clusters, evaluator)
                doc_to_prediction[doc_key] = predicted_clusters

        p, r, f = evaluator.get_prf()
        metrics = {'Eval_Avg_Precision': p * 100, 'Eval_Avg_Recall': r * 100, 'Eval_Avg_F1': f * 100}
        for name, score in metrics.items():
            logger.info('{}: {:.2f}'.format(name, score))
            if tb_writer:
                tb_writer.add_scalar(name, score, step)

        if tb_writer:
            for action_idx, action_count in mention_action_counts.items():
                tb_writer.add_scalar('Mention_Action_Count_{}'.format(action_idx), action_count / total_action_len, step)

        if official:
            conll_results = conll.evaluate_conll(conll_path, doc_to_prediction, stored_info['subtoken_maps'])
            official_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
            logger.info('Official avg F1: {:.4f}'.format(official_f1))

        return f * 100, metrics

    def predict(self, model, tensor_examples):
        logger.info('Predicting {} samples...'.format(len(tensor_examples)))
        model.to(self.device)
        model.eval()

        results = []
        for i, (doc_key, tensor_example) in enumerate(tensor_examples):
            gold_coref_actions = tensor_example[9].tolist()
            tensor_example = tensor_example[:7]
            example_gpu = [d.to(self.device) for d in tensor_example]
            with torch.no_grad():
                model.reset_state()
                predicted_mention_actions, predicted_clusters, predicted_coref_actions, _ = model(*example_gpu)

            doc_preds = {'predicted_mention_actions': predicted_mention_actions,
                         'predicted_clusters': predicted_clusters,
                         'gold_coref_actions': gold_coref_actions,
                         'predicted_coref_actions': predicted_coref_actions}
            results.append(doc_preds)
        return results

    def get_optimizer(self, model):
        # WARNING: check 'layer norm' or 'bias' is not spelled differently or this filter will fail silently!
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        bert_param, task_param = model.get_params(named=True)
        grouped_bert_param = [
            {
                'params': [p for n, p in bert_param if not any(nd in n for nd in no_decay)],
                'lr': self.config['bert_learning_rate'],
                'weight_decay': self.config['adam_weight_decay']
            }, {
                'params': [p for n, p in bert_param if any(nd in n for nd in no_decay)],
                'lr': self.config['bert_learning_rate'],
                'weight_decay': 0.0
            }
        ]
        optimizers = [
            AdamW(grouped_bert_param, lr=self.config['bert_learning_rate'], eps=self.config['adam_eps']),
            Adam(model.get_params()[1], lr=self.config['task_learning_rate'], eps=self.config['adam_eps'], weight_decay=0)
        ]
        return optimizers

    def get_scheduler(self, optimizers, total_update_steps):
        # Only warm up bert lr
        warmup_steps = int(total_update_steps * self.config['warmup_ratio'])

        def lr_lambda_bert(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0, float(total_update_steps - current_step) / float(max(1, total_update_steps - warmup_steps))
            )

        def lr_lambda_task(current_step):
            return max(0.0, float(total_update_steps - current_step) / float(max(1, total_update_steps)))

        schedulers = [
            LambdaLR(optimizers[0], lr_lambda_bert),
            LambdaLR(optimizers[1], lr_lambda_task)
        ]
        return schedulers

    def save_model_checkpoint(self, model, step):
        if step < 30000:
            return  # Don't save early checkpoints
        path_ckpt = join(self.config['log_dir'], f'model_{self.name_suffix}_{step}.bin')
        torch.save(model.state_dict(), path_ckpt)
        logger.info('Saved model to {}'.format(path_ckpt))

    def load_model_checkpoint(self, model, suffix):
        path_ckpt = join(self.config['log_dir'], f'model_{suffix}.bin')
        model.load_state_dict(torch.load(path_ckpt, map_location=torch.device('cpu')), strict=False)
        logger.info('Loaded model from {}'.format(path_ckpt))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument("--model_ckpt", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()

    runner = Runner(args.config_name, args.gpu)
    runner.train(saved_suffix=args.model_ckpt)
