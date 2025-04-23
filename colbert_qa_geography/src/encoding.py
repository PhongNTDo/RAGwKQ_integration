import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
from src.utils import logging


class ColBERTEncoder:
    def __init__(self, checkpoint, query_maxlen, doc_maxlen, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint).to(self.device)
        self.model.eval()
        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen

        self.query_token = "[Q]"
        self.doc_token = "[D]"
        self.mask_token = self.tokenizer.mask_token

    def _encode(self, texts, is_query):
        if is_query:
            texts = [self.query_token + " " + text for text in texts]
            max_len = self.query_maxlen

            inputs = self.tokenizer(texts, return_tensors='pt', padding='max_length', truncation=True, max_length=max_len).to(self.device)
        else:
            texts = [self.doc_token + " " + text for text in texts]
            max_len = self.doc_maxlen
            inputs = self.tokenizer(texts, return_tensors='pt', padding='max_length', truncation=True, max_length=max_len).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs).last_hidden_state

        embeddings = outputs[inputs['attention_mask'].bool()].cpu()

        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        mask = inputs['attention_mask'].bool().cpu()
        return embeddings.numpy(), mask.numpy()