import torch
from torch import nn

class AttentionPooling(nn.Module):
    def __init__(self, input_size, hidden_size, initializer_range=None):
        super(AttentionPooling, self).__init__()
        self.initializer_range = initializer_range
        self.att_fc1 = nn.Linear(input_size, hidden_size)
        self.att_fc2 = nn.Linear(hidden_size, 1)
        self.apply(self.init_weights)
        
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x, attn_mask=None):
        bz = x.shape[0]
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)
        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)
        x = torch.bmm(x.permute(0, 2, 1), alpha)
        x = torch.reshape(x, (bz, -1))
        
        return x


class ResPredictor(nn.Module):
    def __init__(self, input_size, output_size, initializer_range=None) -> None:
        super().__init__()
        self.initializer_range = initializer_range
        self.fc1 = nn.Linear(input_size, input_size)
        self.drop = nn.Dropout(0.1)
        self.fc2 = nn.Linear(input_size, output_size)

        self.apply(self.init_weights)
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def forward(self, weighted_doc_emb, weighted_qry_emb):

        sub_emb = torch.abs(weighted_doc_emb-weighted_qry_emb)
        max_emb = torch.square(torch.max(torch.stack([weighted_doc_emb, weighted_qry_emb], dim=1), dim=1)[0])
        x = torch.cat([weighted_doc_emb, weighted_qry_emb, sub_emb, max_emb], dim=-1)
        y = self.fc1(x)
        y = torch.relu(y)
        y = self.drop(y)
        y = self.fc2(y + x)
        return y
