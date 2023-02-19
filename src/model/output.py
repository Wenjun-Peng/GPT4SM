from dataclasses import dataclass
from transformers.file_utils import ModelOutput
from typing import Optional
import torch

@dataclass
class SiameseOutput(ModelOutput):
    loss: Optional[torch.FloatTensor]=None
    contrastive_loss: Optional[torch.FloatTensor]=None
    ce_loss: Optional[torch.FloatTensor]=None
    ce_logits: Optional[torch.FloatTensor]=None
    encoded_text1: Optional[torch.FloatTensor]=None
    encoded_text2: Optional[torch.FloatTensor]=None