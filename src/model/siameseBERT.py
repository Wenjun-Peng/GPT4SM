from distutils.command.config import config
from lib2to3.pytree import Base
from re import T
from typing import Optional
import os
from numpy import False_
import torch
import torch.nn.functional as F
import copy
from transformers import AutoModelForSequenceClassification, AutoTokenizer,\
    PreTrainedModel, PreTrainedTokenizer, AutoModel, PreTrainedModel
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


# base class for pretrained init 
class BaseBERT(PreTrainedModel):
    def __init__(self, bert, args):
        super().__init__(bert.config)
        self.encoder = bert
        self.args = args
        self.config = bert.config

        if args.data_name == 'blue':
            # self.loss_func = nn.KLDivLoss(reduction='batchmean')
            self.loss_func = nn.CrossEntropyLoss(reduction='mean')
            self.eval_loss_func = nn.CrossEntropyLoss(reduction='mean')
        else:
            if self.config.num_labels == 1:
                self.loss_func = nn.BCEWithLogitsLoss(reduction='mean')
            else:
                self.loss_func = nn.CrossEntropyLoss(reduction='mean')

    @classmethod
    def from_pretrained(cls, args, config):
        bert = AutoModel.from_pretrained(args.model_name_or_path,
                                         from_tf=bool(".ckpt" in args.model_name_or_path),
                                         config=config,
                                         ignore_mismatched_sizes=args.ignore_mismatched_sizes)

        return cls(bert, args)

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
            if self.encoder.config.num_labels == 1:
                ce_logits = ce_logits.view(-1)
            else:
                labels = labels.long()
            ce_loss = self.loss_func(ce_logits, labels)
        return ce_loss, ce_logits


class VanillaBERT(BaseBERT):
    def __init__(self, bert, args):
        super().__init__(bert, args)

        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size*2, args.cls_hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(args.cls_hidden_size, self.encoder.config.num_labels)
        )
    
    def forward(self, text1, text2, labels):
        encoded_text1 = self.encoder(**text1, return_dict=True).pooler_output
        encoded_text2 = self.encoder(**text2, return_dict=True).pooler_output
        # [B x dim] [B x dim] -> [B x dim*2]
        encoded_text = torch.cat([encoded_text1, encoded_text2], dim=-1)
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
                encoded_text1=encoded_text1,
                encoded_text2=encoded_text2
            )


class ContrastiveBERT(BaseBERT):
    def __init__(self, bert, args):
        super().__init__(bert, args)

        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size*2, args.cls_hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(args.cls_hidden_size, self.encoder.config.num_labels)
        )
    
    def forward(self, text1, text2, labels):
        encoded_text1 = self.encoder(**text1, return_dict=True).pooler_output
        encoded_text2 = self.encoder(**text2, return_dict=True).pooler_output
        # [B x dim] [B x dim] -> [B x dim*2]
        encoded_text = torch.cat([encoded_text1, encoded_text2], dim=-1)
        ce_logits = self.classifier(encoded_text)

        # [N x dim] [dim x N] -> [N x N]
        contrastive_logits = torch.matmul(encoded_text1[labels==1], encoded_text2[labels==1].t())

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

        if self.args.data_name == 'blue':
            if self.training:
                ce_logits = F.log_softmax(ce_logits, dim=-1)
                ce_loss = self.loss_func(ce_logits, labels)
            else:
                labels = labels.long()
                ce_loss = self.eval_loss_func(ce_logits, labels)
        else:
            if self.encoder.config.num_labels == 1:
                ce_logits = ce_logits.view(-1)
            else:
                labels = labels.long()
            ce_loss = self.loss_func(ce_logits, labels)

        loss = contrastive_loss + ce_loss

        if self.training:
            return SiameseOutput(
                loss=loss,
                contrastive_loss=contrastive_loss,
                ce_loss=ce_loss,
            )
        else:
            return SiameseOutput(
                loss=loss,
                contrastive_loss=contrastive_loss,
                ce_loss=ce_loss,
                ce_logits=ce_logits,
                encoded_text1=encoded_text1,
                encoded_text2=encoded_text2
            )


class AdditiveBERT(BaseBERT):
    def __init__(self, bert, args):
        super().__init__(bert, args)

        self.pooler = AttentionPooling(
            input_size=bert.config.hidden_size, 
            hidden_size=bert.config.hidden_size, 
            initializer_range=bert.config.initializer_range
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size*2, args.cls_hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(args.cls_hidden_size, self.encoder.config.num_labels)
        )
    
    def forward(self, text1, text2, labels):
        encoded_text1 = self.encoder(**text1, return_dict=True).last_hidden_state
        encoded_text2 = self.encoder(**text2, return_dict=True).last_hidden_state

        encoded_text1 = self.pooler(encoded_text1)
        encoded_text2 = self.pooler(encoded_text2)
        # [B x dim] [B x dim] -> [B x dim*2]
        encoded_text = torch.cat([encoded_text1, encoded_text2], dim=-1)
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
                encoded_text1=encoded_text1,
                encoded_text2=encoded_text2
            )


class AdditiveBERTwithGPTEmb(BaseBERT):
    def __init__(self, bert, args):
        super().__init__(bert, args)

        self.pooler = AttentionPooling(
            input_size=bert.config.hidden_size+args.match_output_size, 
            hidden_size=bert.config.hidden_size, 
            initializer_range=bert.config.initializer_range
        )

        self.transform = nn.Sequential(
            nn.Linear(args.gpt_emb_dim, args.match_hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(args.match_hidden_size, args.match_output_size),
        )

        self.norm = nn.LayerNorm(bert.config.hidden_size+args.match_output_size)

        if self.args.cross_gpt_emb:
            # self.classifier = ResPredictor(
            #     input_size=bert.config.hidden_size*6+args.match_output_size*6,
            #     output_size=self.encoder.config.num_labels,
            #     initializer_range=bert.config.initializer_range
            # )
            # self.cross_norm = nn.LayerNorm(bert.config.hidden_size*2+args.match_output_size)
            self.classifier = nn.Sequential(
                nn.Linear(bert.config.hidden_size*4+args.match_output_size*2, args.cls_hidden_size),
                nn.ReLU(),
                nn.Dropout(args.dropout_rate),
                nn.Linear(args.cls_hidden_size, self.encoder.config.num_labels)
            )

        else:
            self.classifier = nn.Sequential(
                nn.Linear(bert.config.hidden_size*2+args.match_output_size*2, args.cls_hidden_size),
                nn.ReLU(),
                nn.Dropout(args.dropout_rate),
                nn.Linear(args.cls_hidden_size, self.encoder.config.num_labels)
            )

    def forward(self, text1, text2, text1_emb, text2_emb, labels):
        text1_emb = self.transform(text1_emb)
        text2_emb = self.transform(text2_emb)
        if self.args.additive_pooling:
            encoded_text1 = self.encoder(**text1, return_dict=True).last_hidden_state
            encoded_text2 = self.encoder(**text2, return_dict=True).last_hidden_state

            if self.args.cross_gpt_emb:
                text1_emb_qq = text1_emb.unsqueeze(1).repeat([1, encoded_text1.size(1), 1])
                text2_emb_kk = text2_emb.unsqueeze(1).repeat([1, encoded_text2.size(1), 1])
                pooler_input1 = torch.cat([encoded_text1, text1_emb_qq], dim=-1)
                pooler_input2 = torch.cat([encoded_text2, text2_emb_kk], dim=-1)

                text1_emb_qk = text1_emb.unsqueeze(1).repeat([1, encoded_text2.size(1), 1])
                text2_emb_kq = text2_emb.unsqueeze(1).repeat([1, encoded_text1.size(1), 1])
                pooler_input1_cross = torch.cat([encoded_text1, text2_emb_kq], dim=-1)
                pooler_input2_cross = torch.cat([encoded_text2, text1_emb_qk], dim=-1)

                pooler_input1 = self.norm(pooler_input1)
                pooler_input2 = self.norm(pooler_input2)
                pooler_input1_cross = self.norm(pooler_input1_cross)
                pooler_input2_cross = self.norm(pooler_input2_cross)

                encoded_text1 = self.pooler(pooler_input1)
                encoded_text2 = self.pooler(pooler_input2)
                encoded_text1_cross = self.pooler(pooler_input1_cross)
                encoded_text2_cross = self.pooler(pooler_input2_cross)

                encoded_text = torch.cat(
                    [encoded_text1[:,:self.config.hidden_size], 
                     encoded_text2[:,:self.config.hidden_size],
                     encoded_text1_cross[:,:self.config.hidden_size], 
                     encoded_text2_cross[:,:self.config.hidden_size],
                     text1_emb,
                     text2_emb
                    ], dim=-1)
                
                # encoded_text1_cat = torch.cat([encoded_text1[:,:self.config.hidden_size], encoded_text1_cross[:,:self.config.hidden_size], text1_emb], dim=-1)
                # encoded_text2_cat = torch.cat([encoded_text2[:,:self.config.hidden_size], encoded_text2_cross[:,:self.config.hidden_size], text2_emb], dim=-1)
                
                # encoded_text1_cat = self.cross_norm(encoded_text1_cat)
                # encoded_text2_cat = self.cross_norm(encoded_text2_cat)

                # ce_logits = self.classifier(encoded_text1_cat, encoded_text2_cat)
                ce_logits = self.classifier(encoded_text)

            else:
                text1_emb = text1_emb.unsqueeze(1).repeat([1, encoded_text1.size(1), 1])
                text2_emb = text2_emb.unsqueeze(1).repeat([1, encoded_text2.size(1), 1])
                pooler_input1 = torch.cat([encoded_text1, text1_emb], dim=-1)
                pooler_input2 = torch.cat([encoded_text2, text2_emb], dim=-1)

                pooler_input1 = self.norm(pooler_input1)
                pooler_input2 = self.norm(pooler_input2)
                encoded_text1 = self.pooler(pooler_input1)
                encoded_text2 = self.pooler(pooler_input2)
                encoded_text = torch.cat([encoded_text1, encoded_text2], dim=-1)
                ce_logits = self.classifier(encoded_text)
        else:
            encoded_text1 = self.encoder(**text1, return_dict=True).pooler_output
            encoded_text2 = self.encoder(**text2, return_dict=True).pooler_output
            encoded_text1 = torch.cat([encoded_text1, text1_emb], dim=-1)
            encoded_text2 = torch.cat([encoded_text2, text2_emb], dim=-1)
            encoded_text = torch.cat([encoded_text1, encoded_text2], dim=-1)
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
                encoded_text1=encoded_text1,
                encoded_text2=encoded_text2
            )


class BERTwithGPTEmbToken(AdditiveBERT):
    def __init__(self, bert, args):
        super().__init__(bert, args)
        self.transform = nn.Sequential(
            nn.Linear(args.gpt_emb_dim, args.match_hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(args.match_hidden_size, self.config.hidden_size),
        )


    def forward(self, text1, text2, text1_emb, text2_emb, labels):
        # B x 1 x dim
        text1_emb = self.transform(text1_emb).unsqueeze(1)
        text2_emb = self.transform(text2_emb).unsqueeze(1)

        # B x L x dim
        text1_bert_emb = self.encoder.embeddings.word_embeddings(text1["input_ids"])
        text2_bert_emb = self.encoder.embeddings.word_embeddings(text2["input_ids"])

        # B x (L+1) x dim
        text1_input_emb = torch.cat([text1_emb, text1_bert_emb], dim=1)
        text2_input_emb = torch.cat([text2_emb, text2_bert_emb], dim=1)

        gpt_emb_mask = torch.ones((text1["attention_mask"].size(0), 1), dtype=torch.long, device=text1["attention_mask"].device)

        text1_attention_mask = torch.cat([gpt_emb_mask, text1["attention_mask"]], dim=1)
        text2_attention_mask = torch.cat([gpt_emb_mask, text2["attention_mask"]], dim=1)
        
        input1 = {"inputs_embeds": text1_input_emb, "attention_mask": text1_attention_mask}
        input2 = {"inputs_embeds": text2_input_emb, "attention_mask": text2_attention_mask}

        if self.args.additive_pooling:
            encoded_text1 = self.encoder(**input1, return_dict=True).last_hidden_state
            encoded_text2 = self.encoder(**input2, return_dict=True).last_hidden_state
            encoded_text1 = self.pooler(encoded_text2)
            encoded_text2 = self.pooler(encoded_text2)
        else:
            encoded_text1 = self.encoder(**input1, return_dict=True).pooler_output
            encoded_text2 = self.encoder(**input2, return_dict=True).pooler_output

        # [B x dim] [B x dim] -> [B x dim*2]
        encoded_text = torch.cat([encoded_text1, encoded_text2], dim=-1)
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
                encoded_text1=encoded_text1,
                encoded_text2=encoded_text2
            )



class BERTwithGPTRegCopy(BaseBERT):
    def __init__(self, bert, args):
        super().__init__(bert, args)
        self.copy_loss_func = nn.MSELoss(reduction='none')

        self.pooler = AttentionPooling(
            input_size=self.args.gpt_emb_dim, 
            hidden_size=self.config.hidden_size, 
            initializer_range=self.config.initializer_range
        )

        self.transform = nn.Sequential(
            nn.Linear(self.config.hidden_size, args.match_hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(args.match_hidden_size, self.args.gpt_emb_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.args.gpt_emb_dim*2, args.cls_hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(args.cls_hidden_size, self.encoder.config.num_labels)
        )

        self.norm = torch.nn.LayerNorm(self.args.gpt_emb_dim)
    
    def forward(self, text1, text2, text1_emb, text2_emb, labels):
        output1 = self.encoder(**text1, return_dict=True)
        output2 = self.encoder(**text2, return_dict=True)
        encoded_text1 = output1.last_hidden_state
        encoded_text2 = output2.last_hidden_state

        pooler_text1 = self.transform(output1.pooler_output)
        pooler_text2 = self.transform(output2.pooler_output)

        pooler_text1 = pooler_text1 / torch.norm(pooler_text1, p=2, dim=1, keepdim=True)
        pooler_text2 = pooler_text2 / torch.norm(pooler_text2, p=2, dim=1, keepdim=True)
            
        encoded_text1 = self.transform(encoded_text1)
        encoded_text2 = self.transform(encoded_text2)
        
        encoded_text1 = self.pooler(encoded_text1)
        encoded_text2 = self.pooler(encoded_text2)
        
        res_encoded_text1 = self.norm(encoded_text1+pooler_text1)
        res_encoded_text2 = self.norm(encoded_text2+pooler_text2)
        # [B x dim] [B x dim] -> [B x dim*2]
        encoded_text = torch.cat([res_encoded_text1, res_encoded_text2], dim=-1)
        ce_logits = self.classifier(encoded_text)

        if self.reg_copy_mode == 'reg':
            copy_loss = self.args.contrastive_weight * (self.copy_loss_func(pooler_text1, text1_emb).sum(-1).mean() + \
                                                        self.copy_loss_func(pooler_text1, text2_emb).sum(-1).mean()) / 2
            ce_loss, ce_logits = self.compute_loss(ce_logits, labels)
            loss = ce_loss + copy_loss
        elif self.reg_copy_mode == 'copy':
            loss = (self.copy_loss_func(pooler_text1, text1_emb).sum(-1).mean() + \
                        self.copy_loss_func(pooler_text1, text2_emb).sum(-1).mean()) / 2
            ce_loss = None
            copy_loss = None

        elif self.reg_copy_mode == 'cls':
            loss, ce_logits = self.compute_loss(ce_logits, labels)
            ce_loss = None
            copy_loss = None
        
        else:
            loss = None
            ce_loss = None
            copy_loss = None



        if self.training:
            return SiameseOutput(
                loss=loss,
                ce_loss=ce_loss,
                contrastive_loss=copy_loss
            )
        else:
            return SiameseOutput(
                loss=loss,
                ce_loss=ce_loss,
                contrastive_loss=copy_loss,
                ce_logits=ce_logits,
                encoded_text1=encoded_text1,
                encoded_text2=encoded_text2
            )
        

class BERTwithGPTRegCopyEmb(BaseBERT):
    def __init__(self, bert, args):
        super().__init__(bert, args)
        self.copy_loss_func = nn.MSELoss(reduction='none')

        self.pooler = AttentionPooling(
            input_size=self.config.hidden_size*2, 
            hidden_size=self.config.hidden_size, 
            initializer_range=self.config.initializer_range
        )

        self.transform = nn.Sequential(
            nn.Linear(self.config.hidden_size, args.match_hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(args.match_hidden_size, self.args.gpt_emb_dim),
        )

        self.transform2 = nn.Sequential(
            nn.Linear(self.args.gpt_emb_dim, args.match_hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(args.match_hidden_size, self.config.hidden_size),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size*4, args.cls_hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(args.cls_hidden_size, self.encoder.config.num_labels)
        )

        self.norm = torch.nn.LayerNorm(self.config.hidden_size*2)
    
    def forward(self, text1, text2, text1_emb, text2_emb, labels):
        output1 = self.encoder(**text1, return_dict=True)
        output2 = self.encoder(**text2, return_dict=True)
        encoded_text1 = output1.last_hidden_state
        encoded_text2 = output2.last_hidden_state

        pooler_text1 = self.transform(output1.pooler_output)
        pooler_text2 = self.transform(output2.pooler_output)

        copied_text1 = pooler_text1 / torch.norm(pooler_text1, p=2, dim=1, keepdim=True)
        copied_text2 = pooler_text2 / torch.norm(pooler_text2, p=2, dim=1, keepdim=True)
            
        copied_text1_emb = self.transform2(pooler_text1)
        copied_text2_emb = self.transform2(pooler_text2)
        
        copied_text1_emb = copied_text1_emb.unsqueeze(1).repeat([1, encoded_text1.size(1), 1])
        copied_text2_emb = copied_text2_emb.unsqueeze(1).repeat([1, encoded_text2.size(1), 1])
        pooler_input1 = torch.cat([encoded_text1, copied_text1_emb], dim=-1)
        pooler_input2 = torch.cat([encoded_text2, copied_text2_emb], dim=-1)

        pooler_input1 = self.norm(pooler_input1)
        pooler_input2 = self.norm(pooler_input2)
        encoded_text1 = self.pooler(pooler_input1)
        encoded_text2 = self.pooler(pooler_input2)
        encoded_text = torch.cat([encoded_text1, encoded_text2], dim=-1)
        ce_logits = self.classifier(encoded_text)

        if self.reg_copy_mode == 'reg':
            copy_loss = self.args.contrastive_weight * (self.copy_loss_func(copied_text1, text1_emb).sum(-1).mean() + \
                                                        self.copy_loss_func(copied_text2, text2_emb).sum(-1).mean()) / 2
            ce_loss, ce_logits = self.compute_loss(ce_logits, labels)
            loss = ce_loss + copy_loss
        elif self.reg_copy_mode == 'copy':
            loss = (self.copy_loss_func(copied_text1, text1_emb).sum(-1).mean() + \
                    self.copy_loss_func(copied_text2, text2_emb).sum(-1).mean()) / 2
            ce_loss = None
            copy_loss = None

        elif self.reg_copy_mode == 'cls':
            loss, ce_logits = self.compute_loss(ce_logits, labels)
            ce_loss = None
            copy_loss = None
        
        else:
            loss = None
            ce_loss = None
            copy_loss = None



        if self.training:
            return SiameseOutput(
                loss=loss,
                ce_loss=ce_loss,
                contrastive_loss=copy_loss
            )
        else:
            return SiameseOutput(
                loss=loss,
                ce_loss=ce_loss,
                contrastive_loss=copy_loss,
                ce_logits=ce_logits,
                encoded_text1=encoded_text1,
                encoded_text2=encoded_text2
            )


class BERTwith0Epoch(BaseBERT):
    def __init__(self, bert, args):
        super().__init__(bert, args)
        self.copy_loss_func = nn.MSELoss(reduction='none')

        self.pooler = AttentionPooling(
            input_size=self.config.hidden_size*2, 
            hidden_size=self.config.hidden_size, 
            initializer_range=self.config.initializer_range
        )

        self.transform = nn.Sequential(
            nn.Linear(self.config.hidden_size, args.match_hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(args.match_hidden_size, self.args.gpt_emb_dim),
        )

        self.transform2 = nn.Sequential(
            nn.Linear(self.args.gpt_emb_dim, args.match_hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(args.match_hidden_size, self.config.hidden_size),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size*4, args.cls_hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(args.cls_hidden_size, self.encoder.config.num_labels)
        )

        self.norm = torch.nn.LayerNorm(self.config.hidden_size*2)
    
    def forward(self, text1, text2, text1_emb, text2_emb, labels):
        output1 = self.encoder(**text1, return_dict=True)
        output2 = self.encoder(**text2, return_dict=True)
        encoded_text1 = output1.last_hidden_state
        encoded_text2 = output2.last_hidden_state

        pooler_text1 = self.transform(output1.pooler_output)
        pooler_text2 = self.transform(output2.pooler_output)
            
        copied_text1_emb = self.transform2(pooler_text1)
        copied_text2_emb = self.transform2(pooler_text2)
        
        copied_text1_emb = copied_text1_emb.unsqueeze(1).repeat([1, encoded_text1.size(1), 1])
        copied_text2_emb = copied_text2_emb.unsqueeze(1).repeat([1, encoded_text2.size(1), 1])
        pooler_input1 = torch.cat([encoded_text1, copied_text1_emb], dim=-1)
        pooler_input2 = torch.cat([encoded_text2, copied_text2_emb], dim=-1)

        pooler_input1 = self.norm(pooler_input1)
        pooler_input2 = self.norm(pooler_input2)
        encoded_text1 = self.pooler(pooler_input1)
        encoded_text2 = self.pooler(pooler_input2)
        encoded_text = torch.cat([encoded_text1, encoded_text2], dim=-1)
        ce_logits = self.classifier(encoded_text)

        loss, ce_logits = self.compute_loss(ce_logits, labels)
        ce_loss = None
        copy_loss = None

        if self.training:
            return SiameseOutput(
                loss=loss,
                ce_loss=ce_loss,
                contrastive_loss=copy_loss
            )
        else:
            return SiameseOutput(
                loss=loss,
                ce_loss=ce_loss,
                contrastive_loss=copy_loss,
                ce_logits=ce_logits,
                encoded_text1=encoded_text1,
                encoded_text2=encoded_text2
            )