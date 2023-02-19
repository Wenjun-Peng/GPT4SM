from distutils.command.config import config
from re import T
from typing import Optional
import os
from numpy import False_
import torch
import torch.nn.functional as F
import copy
from transformers import AutoModelForSequenceClassification, AutoTokenizer,\
    PreTrainedModel, PreTrainedTokenizer, BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertConfig
from transformers.models.longformer import modeling_longformer
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPooling
from torch import nn
import torch.distributed as dist
import logging
import copy
from transformers.file_utils import ModelOutput

from transformers.file_utils import (
    WEIGHTS_NAME,
    is_torch_tpu_available,
)
from model.output import SiameseOutput
from model.layers import AttentionPooling, ResPredictor


class BaseGPT(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.args = args
        self.config = config
        if args.data_name == 'blue':
            self.loss_func = nn.KLDivLoss(reduction='batchmean')
            self.eval_loss_func = nn.CrossEntropyLoss(reduction='mean')
        else:
            if self.config.num_labels == 1:
                self.loss_func = nn.BCEWithLogitsLoss(reduction='mean')
            else:
                self.loss_func = nn.CrossEntropyLoss(reduction='mean')

    @classmethod
    def from_pretrained(cls, args, config):
        return cls(config, args)

    def compute_loss(self, ce_logits, labels):
        if self.args.data_name == 'blue':
            if self.training:
                # ls_ce_logits = F.log_softmax(ce_logits, dim=-1)
                # ce_loss = self.loss_func(ls_ce_logits, labels)
                labels = labels.long()
                ce_loss = self.eval_loss_func(ce_logits, labels)
            else:
                labels = labels.long()
                ce_loss = self.eval_loss_func(ce_logits, labels)
        else:
            if self.config.num_labels == 1:
                ce_logits = ce_logits.view(-1)
            else:
                labels = labels.long()
            ce_loss = self.loss_func(ce_logits, labels)
        return ce_loss, ce_logits


class VanillaGPT(BaseGPT):
    def __init__(self, config, args):
        super().__init__(config, args)

        self.classifier = nn.Sequential(
            nn.Linear(args.gpt_emb_dim*2, args.cls_hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(args.cls_hidden_size, self.config.num_labels)
        )
    
    def forward(self, text1, text2, text1_emb, text2_emb, labels):
        # [B x dim] [B x dim] -> [B x dim*2]
        encoded_text = torch.cat([text1_emb, text2_emb], dim=-1)
        ce_logits = self.classifier(encoded_text)

        ce_loss, ce_logits = self.compute_loss(ce_logits, labels)

        if self.training:
            return SiameseOutput(
                loss=ce_loss,
            )
        else:
            return SiameseOutput(
                loss=ce_loss,
                ce_logits=ce_logits,
                encoded_text1=text1_emb,
                encoded_text2=text2_emb
            )


class CosGPT(BaseGPT):
    def __init__(self, config, args):
        super().__init__(config, args)
    
    def forward(self, text1, text2, text1_emb, text2_emb, labels):
        # [B x dim] [B x dim] -> [B x dim*2]
        ce_logits = torch.sum(torch.mul(text1_emb, text2_emb), dim=-1)
        loss = torch.zeros(1, dtype=torch.float, device=labels.device)
        # print(ce_logits)
        return SiameseOutput(
            loss=loss,
            ce_logits=ce_logits,
            encoded_text1=text1_emb,
            encoded_text2=text2_emb
        )


class ContrastiveGPT(BaseGPT):
    def __init__(self, config, args):
        super().__init__(config, args)

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.transform = nn.Sequential(
            nn.Linear(args.gpt_emb_dim, args.match_hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(args.match_hidden_size, args.match_output_size),
        )

        self.classifier = nn.Sequential(
            nn.Linear(args.gpt_emb_dim*2, args.cls_hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(args.cls_hidden_size, self.config.num_labels)
        )
    
    def forward(self, text1, text2, text1_emb, text2_emb, labels):
        transformed_text1 = self.transform(text1_emb)
        transformed_text2 = self.transform(text2_emb)
        
        # [B x dim] [B x dim] -> [B x dim*2]
        encoded_text = torch.cat([text1_emb, text2_emb], dim=-1)
        ce_logits = self.classifier(encoded_text)

        ce_loss = self.compute_loss(ce_logits, labels)

        # [N x dim] [dim x N] -> [N x N]
        contrastive_logits = torch.matmul(transformed_text1[labels==1], transformed_text2[labels==1].t())

        if self.args.temperature is not None:
            assert self.args.temperature > 0
            contrastive_logits = contrastive_logits / self.args.temperature

        N = torch.sum((labels==1).long())
        if N != 0:
            contrastive_logits = contrastive_logits.view(N, N)
            target_label = torch.arange(0, contrastive_logits.size(0), dtype=torch.long, device=contrastive_logits.device)
            contrastive_loss = self.args.contrastive_weight * self.cross_entropy(contrastive_logits, target_label)
        else:
            contrastive_loss = 0

        ce_loss, ce_logits = self.args.bce_weight * self.loss_func(ce_logits.view(-1), labels)

        loss = contrastive_loss + ce_loss

        if self.training:
            return SiameseOutput(
                loss=loss,
                contrastive_loss=contrastive_loss,
                ce_loss=ce_loss,
            )
        else:
            return SiameseOutput(
                loss=ce_loss,
                contrastive_loss=contrastive_loss,
                ce_loss=ce_loss,
                ce_logits=ce_logits,
                encoded_text1=transformed_text1,
                encoded_text2=transformed_text2
            )


class ResHeadGPT(BaseGPT):
    def __init__(self, config, args):
        super().__init__(config, args)

        self.classifier = ResPredictor(config)


    def forward(self, text1, text2, text1_emb, text2_emb, labels):
        # [B x dim] [B x dim] -> [B x dim*2]
        # encoded_text = torch.cat([text1_emb, text2_emb], dim=-1)
        ce_logits = self.classifier(text1_emb, text2_emb)

        ce_loss, ce_logits = self.compute_loss(ce_logits, labels)

        if self.training:
            return SiameseOutput(
                loss=ce_loss,
            )
        else:
            return SiameseOutput(
                loss=ce_loss,
                ce_logits=ce_logits,
                encoded_text1=text1_emb,
                encoded_text2=text2_emb
            )