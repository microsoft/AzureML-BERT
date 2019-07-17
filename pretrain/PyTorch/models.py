import argparse
import logging
import random
import numpy as np
import os
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn import CrossEntropyLoss, MSELoss
from logger import Logger

from dataset import BatchType
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel, BertConfig
from pytorch_pretrained_bert.modeling import BertPreTrainingHeads, BertPreTrainedModel, BertPreTrainingHeads, BertLMPredictionHead
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE


class BertPretrainingLoss(BertPreTrainedModel):
    def __init__(self, bert_encoder, config):
        super(BertPretrainingLoss, self).__init__(config)
        self.bert = bert_encoder
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight)
        self.cls.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False)
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output)

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        else:
            return prediction_scores, seq_relationship_score


class MTLRouting(nn.Module):
    """This setup is to add MultiTask Training support in BERT Training. 
    """
    def __init__(self, encoder: BertModel, write_log, summary_writer):
        super(MTLRouting, self).__init__()
        self.bert_encoder = encoder
        self._batch_loss_calculation = nn.ModuleDict()
        self._batch_counter = {}
        self._batch_module_name = {}
        self._batch_name = {}
        self.write_log = write_log
        self.logger = Logger(cuda=torch.cuda.is_available())
        self.summary_writer = summary_writer

    def register_batch(self, batch_type, module_name, loss_calculation: nn.Module):
        assert isinstance(loss_calculation, nn.Module)
        self._batch_loss_calculation[str(batch_type.value)] = loss_calculation
        self._batch_counter[batch_type] = 0
        self._batch_module_name[batch_type] = module_name

    def log_summary_writer(self, batch_type, logs: dict, base='Train'):
        if self.write_log:
            counter = self._batch_counter[batch_type]
            module_name = self._batch_module_name.get(
                batch_type, self._get_batch_type_error(batch_type))
            for key, log in logs.items():
                self.summary_writer.add_scalar(
                    f'{base}/{module_name}/{key}', log, counter)
            self._batch_counter[batch_type] = counter + 1

    def _get_batch_type_error(self, batch_type):
        def f(*args, **kwargs):
            message = f'Misunderstood batch type of {batch_type}'
            self.logger.error(message)
            raise ValueError(message)
        return f

    def forward(self, batch, log=True):
        batch_type = batch[0][0].item()

        # Pretrain Batch
        if batch_type == BatchType.PRETRAIN_BATCH:
            loss_function = self._batch_loss_calculation[str(batch_type)]

            loss = loss_function(input_ids=batch[1],
                                 token_type_ids=batch[3],
                                 attention_mask=batch[2],
                                 masked_lm_labels=batch[5],
                                 next_sentence_label=batch[4])
            if log:
                self.log_summary_writer(
                    batch_type, logs={'pretrain_loss': loss.item()})
            return loss


class BertMultiTask:
    def __init__(self, job_config, use_pretrain, tokenizer, cache_dir, device, write_log, summary_writer):
        self.job_config = job_config

        if not use_pretrain:
            model_config = self.job_config.get_model_config()
            bert_config = BertConfig(**model_config)
            bert_config.vocab_size = len(tokenizer.vocab)

            self.bert_encoder = BertModel(bert_config)
        # Use pretrained bert weights
        else:
            self.bert_encoder = BertModel.from_pretrained(self.job_config.get_model_file_type(), cache_dir=cache_dir)
            bert_config = self.bert_encoder.config

        self.network=MTLRouting(self.bert_encoder, write_log = write_log, summary_writer = summary_writer)

        #config_data=self.config['data']

        # Pretrain Dataset
        self.network.register_batch(BatchType.PRETRAIN_BATCH, "pretrain_dataset", loss_calculation=BertPretrainingLoss(self.bert_encoder, bert_config))

        self.device=device
        # self.network = self.network.float()
        # print(f"Bert ID: {id(self.bert_encoder)}  from GPU: {dist.get_rank()}")

    def save(self, filename: str):
        network=self.network.module
        return torch.save(network.state_dict(), filename)

    def load(self, model_state_dict: str):
        return self.network.module.load_state_dict(torch.load(model_state_dict, map_location=lambda storage, loc: storage))

    def move_batch(self, batch, non_blocking=False):
        return batch.to(self.device, non_blocking)

    def eval(self):
        self.network.eval()

    def train(self):
        self.network.train()

    def save_bert(self, filename: str):
        return torch.save(self.bert_encoder.state_dict(), filename)

    def to(self, device):
        assert isinstance(device, torch.device)
        self.network.to(device)

    def half(self):
        self.network.half()
