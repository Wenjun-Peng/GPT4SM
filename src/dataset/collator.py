from transformers import DataCollatorWithPadding
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from collections import defaultdict
import torch

class SiameseBERTCollator(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        text1 = []
        text2 = []
        labels = []
        for f in features:
            text1.append(f['text1'])
            text2.append(f['text2'])
            labels.append(f['labels'])

        if len(text1) != 0:
            text1_batch = self.tokenizer.pad(
                text1,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )

        if len(text1) != 0:
            text2_batch = self.tokenizer.pad(
                text2,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
        
        labels = torch.as_tensor(labels, dtype=torch.float)
        batch = {'text1': text1_batch, 'text2': text2_batch, 'labels': labels}
        return batch


class SiameseGPTCollator(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        text1 = []
        text2 = []
        labels = []
        for f in features:
            text1.append(f['text1_emb'])
            text2.append(f['text2_emb'])
            labels.append(f['labels'])

        
        text1 = torch.as_tensor(text1, dtype=torch.float)
        text2 = torch.as_tensor(text2, dtype=torch.float)
        labels = torch.as_tensor(labels, dtype=torch.float)
        batch = {'text1_emb': text1, 'text2_emb': text2, 'labels': labels}
        return batch


class BERTwithGPTEmbCollator(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        text1 = []
        text2 = []
        text1_emb = []
        text2_emb = []
        labels = []
        for f in features:
            text1.append(f['text1'])
            text2.append(f['text2'])
            text1_emb.append(f['text1_emb'])
            text2_emb.append(f['text2_emb'])
            labels.append(f['labels'])

        if len(text1) != 0:
            text1_batch = self.tokenizer.pad(
                text1,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )

        if len(text2) != 0:
            text2_batch = self.tokenizer.pad(
                text2,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
        
        text1_emb = torch.as_tensor(text1_emb, dtype=torch.float)
        text2_emb = torch.as_tensor(text2_emb, dtype=torch.float)
        labels = torch.as_tensor(labels, dtype=torch.float)
        batch = {'text1': text1_batch, 'text2': text2_batch, 'text1_emb': text1_emb, 'text2_emb': text2_emb, 'labels': labels}
        return batch

