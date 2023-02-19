from torch import nn
from transformers import BertModel, BertPreTrainedModel, BertConfig
import torch
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import logging
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self, return_type=None):
        super().__init__()
        self.return_type = return_type
        # self.num_labels = config.num_labels
        # self.config = config

        # self.bert = BertModel(config)
        # config = BertConfig.from_pretrained("bert-base-cased", output_hidden_states=True)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        classifier_dropout = 0.2
        self.dropout = nn.Dropout(classifier_dropout)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # self.transform_linear = nn.Sequential(
        #     nn.Dropout(classifier_dropout),
        #     nn.Linear(config.hidden_size, config.hidden_size),
        #     nn.BatchNorm1d(config.hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(config.hidden_size, 12288 // 4),
        #     nn.BatchNorm1d(12288 // 4),
        #     nn.ReLU(),
        #     nn.Linear(12288 // 4, 12288),
        #     nn.BatchNorm1d(12288),
        # )

        self.transform_linear = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(768, 768, bias=False),
            nn.ReLU(),
            nn.Linear(768, 1536, bias=False),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(1536, 1536),
            nn.ReLU(),
            nn.Linear(1536, 18),
        )
        

        # # Initialize weights and apply final processing
        # self.post_init()

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                # m.bias.data.fill_(0.01)
        self.transform_linear.apply(init_weights)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        gpt1=None,
        gpt2=None,
        fix_embedding=False,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # print("enter no fix embedding")
        # print("enter base model")
        # print(input_ids.shape)
        # print(input_ids)
        # print(attention_mask.shape)
        # print(attention_mask)
        # print(token_type_ids.shape)
        if not fix_embedding:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                # position_ids=position_ids,
                # head_mask=head_mask,
                # inputs_embeds=inputs_embeds,
                # output_attentions=output_attentions,
                # output_hidden_states=output_hidden_states,
                # return_dict=return_dict,
            )
            last_hidden_state = outputs.last_hidden_state
            # print("last_hidden_state",last_hidden_state.shape)
            # pooled_output = outputs[1]
            pooled_output = self.avgpool(last_hidden_state.permute([0, 2, 1])).permute([0, 2, 1]).squeeze(1)
            # print("last_hidden_state",last_hidden_state.shape)
            pooled_output = self.dropout(pooled_output)
            transform_emb_no_norm = self.transform_linear(pooled_output)
            # print("transform_emb",transform_emb.shape)
            transform_emb = transform_emb_no_norm / torch.norm(transform_emb_no_norm, dim=1).unsqueeze(1)
        else:
            with torch.no_grad():
                outputs = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                last_hidden_state = outputs.last_hidden_state
                # print("last_hidden_state",last_hidden_state.shape)
                # pooled_output = outputs[1]
                pooled_output = self.avgpool(last_hidden_state.permute([0, 2, 1])).permute([0, 2, 1]).squeeze(1)
                pooled_output = self.dropout(pooled_output)
                transform_emb_no_norm = self.transform_linear(pooled_output)
                transform_emb = transform_emb_no_norm / torch.norm(transform_emb_no_norm, dim=1).unsqueeze(1)
            
        
        if self.return_type == 'bert':
            return last_hidden_state
        elif self.return_type == 'emb':
            return transform_emb
        elif self.return_type == 'emb_no_norm':
            return transform_emb_no_norm
        else:
            raise NotImplementedError

    def info_nce(self, emb, gpt1):
        bz, emb_size = emb.shape 
        emb_expand = torch.stack([emb[i].repeat(bz, 1).clone() for i in range(bz)]) # [bz, bz, emb_size]

        """
        [1, 2, 3]     [0.1, 0.2, 0.3]
        [4, 5, 6]     [0.4, 0.5, 0.6] 


        [1, 2, 3]    [0.1, 0.2, 0.3] = 2  0
        [1, 2, 3]    [0.4, 0.5, 0.6] =2 
        
        [4, 5, 6]    [0.1, 0.2, 0.3]
        [4, 5, 6]    [0.4, 0.5, 0.6]  1
        """

        gpt_expand = gpt1.repeat(bz, 1, 1) # [bz, bz, emb_size]
        # mse_loss = MSELoss(reduction="none")
        ce_loss = CrossEntropyLoss()
        # mse_score = mse_loss(emb_expand, gpt_expand) # [bz, bz, emb_size]
        score = torch.mul(emb_expand, gpt_expand).sum(dim=2) # [bz, bz]
        # print("mse_score", mse_score.shape)
        label = torch.tensor(list(range(bz)), dtype=torch.long).to(emb.device)

        loss = ce_loss(score, label)
        return loss

class ForLoad(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model


class GPTClassifier(nn.Module):
    def __init__(self, num_labels, dropout=0.2):
        super().__init__()
        self.num_labels = num_labels
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(12288, 12288),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(12288, num_labels),
        )

    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        gpt1=None,
        gpt2=None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:


        logits = self.classifier(gpt1)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=None,
                attentions=None,
            )