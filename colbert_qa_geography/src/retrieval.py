import time
from collections import defaultdict
import numpy as np
import torch
from src.utils import logging


class ColBERTRetriever:
    def __init__(self, encoder, indexer, passage_map, all_doc_embeddings, embedding_map):
        self.encoder = encoder
        self.indexer = indexer
        self.passage_map = passage_map
        self.all_doc_embeddings = torch.tensor(all_doc_embeddings).to(encoder.device)
        self.embedding_map = embedding_map
        self.passage_embedding_indices = self._map_passage_to_embedding_range(embedding_map)

    def _map_passage_to_embedding_range(self, embedding_map):
        passage_indices = defaultdict(list)
        for idx, p_id in enumerate(embedding_map):
            passage_indices[p_id].append(idx)
        
        passage_ranges = {}
        for p_id, indices in passage_indices.items():
            if indices:
                passage_ranges[p_id] = (min(indices), max(indices) + 1)
        return passage_ranges
    
    def retrieve(self, query_text, candidate_k, top_n):
        start_time = time.time()