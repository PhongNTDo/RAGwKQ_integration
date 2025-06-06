import time
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
from src.utils import logging


class GeoRetriever:
    def __init__(self, encoder, indexer, passage_map, all_doc_embeddings, embedding_map):
        self.encoder = encoder
        self.indexer = indexer
        self.passage_map = passage_map
        self.all_doc_embeddings = all_doc_embeddings
        self.embedding_map = embedding_map

    def _map_passage_to_embedding_range(self, embedding_map):
        passage_indices = defaultdict(list)
        for idx, p_id in enumerate(embedding_map):
            passage_indices[p_id].append(idx)
        
        passage_ranges = {}
        for p_id, indices in passage_indices.items():
            if indices:
                passage_ranges[p_id] = (min(indices), max(indices) + 1)
        return passage_ranges
    
    def retrieve(self, query_text, candidate_k, top_n=-1):
        start_time = time.time()

        # 1. Encode query
        query_embeddings = self.encoder.encode_query(query_text)
        query_encode_time = time.time()
        logging.info(f"Query encoded in {query_encode_time - start_time:.2f} seconds")

        # 2. Candidate retrieval (Faiss search - Stage 1)
        candidate_passage_scores = defaultdict(int)
        distances, indices = self.indexer.search(query_embeddings, candidate_k)
        faiss_search_time = time.time()
        logging.info(f"Faiss search in {faiss_search_time - query_encode_time:.2f} seconds")

        retrieved_passages_ids = set()
        print(indices)
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                embedding_idx = indices[i][j]
                if embedding_idx != -1:
                    passage_id = self.embedding_map[embedding_idx]
                    retrieved_passages_ids.add(passage_id)
                    candidate_passage_scores[passage_id] += distances[i][j]

        candidate_gen_time = time.time()
        logging.info(f"Candidate generation in {candidate_gen_time - faiss_search_time:.2f} seconds")

        if not retrieved_passages_ids:
            logging.warning("No candidate passages found.")
            return []
        
        # 3. Re-ranking
        # Pass (coming soon)
        
        # 4. Post-processing
        sorted_passages = sorted(candidate_passage_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for i, (passage_id, score) in enumerate(sorted_passages):
            passage_id, score = sorted_passages[i]
            results.append({
                'passage_id': passage_id,
                'score': score,
                'text': self.passage_map.get(passage_id, "Text not found")
            })

        total_time = time.time() - start_time
        logging.info(f"Total retrieval time: {total_time:.2f} seconds")
        return results