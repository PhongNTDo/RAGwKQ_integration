import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
# from src.utils import logging

class Encoder:
    def __init__(self, checkpoint, query_maxlen, doc_maxlen, device='cuda'):
        # self.model = BGEM3FlagModel(checkpoint, 
        #                             query_max_length=query_maxlen,
        #                             passage_max_length=doc_maxlen,
        #                             query_instruction_for_retrieval="Repersent this sentence for searching relevant passages:",
        #                             use_fp16=True)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint)
        self.model.to(device)

        self.doc_maxlen = doc_maxlen
        self.query_maxlen = query_maxlen
        self.query_instruction_for_retrieval = "Represent this sentence for searching relevant passages:"
        self.device = device

    def _encode(self, texts, is_query):
        if is_query:
            texts = [self.query_instruction_for_retrieval + text for text in texts]
            max_length = self.query_maxlen
            inputs = self.tokenizer(texts, return_tensors="pt", padding='max_length', truncation=True, max_length=max_length).to(self.device)

        else:
            max_length = self.doc_maxlen
            inputs = self.tokenizer(texts, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs).last_hidden_state
            embeddings = torch.nn.functional.normalize(outputs[:, 0, :], p=2, dim=1)
        return embeddings


    def encode_passages_batch(self, passages_batch):
        embedding_list = []
        passages_embedding = self._encode(passages_batch, is_query=False)
        for i in range(passages_embedding.shape[0]):
            embedding_list.append(passages_embedding[i].cpu().numpy())
        return embedding_list

    def encode_query(self, query_text):
        query_embedding = self._encode(query_text, is_query=True)
        return query_embedding.cpu().numpy()


if __name__ == '__main__':
    encoder = Encoder(checkpoint="BAAI/bge-m3", query_maxlen=128, doc_maxlen=1024)
    corpus = ["Japan[a] is an island country in East Asia. Located in the Pacific Ocean off the northeast coast of the Asian mainland, it is bordered on the west by the Sea of Japan and extends from the Sea of Okhotsk in the north to the East China Sea in the south.", 
    "The capital of Japan and its largest city is Tokyo; the Greater Tokyo Area is the largest metropolitan area in the world, with more than 37 million inhabitants as of 2024.",
    "Japan is divided into 47 administrative prefectures and eight traditional regions."]
    query = "What is the capital of Japan?"
    passages_embedding = encoder.encode_passages_batch(corpus)
    query_embedding = encoder.encode_query([query])
    print(len(passages_embedding))
    print(passages_embedding[0].shape)
    print(query_embedding.shape)