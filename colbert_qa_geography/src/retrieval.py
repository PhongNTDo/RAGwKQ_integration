import time
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
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

        # 1. Encode query
        query_embeddings = self.encoder.encode_query(query_text) # (query_malex, dim)
        query_encode_time = time.time()
        logging.info(f"Query encoded in {query_encode_time - start_time:.4f}s")

        # 2. Candidate retrieval (Faiss search - Stage 1)
        candidate_passage_scores = defaultdict(int)
        distances, indices = self.indexer.search(query_embeddings, candidate_k)
        faiss_search_time = time.time()
        logging.info(f"Faiss search in {faiss_search_time - query_encode_time:.4f}s")

        retrieved_passage_ids = set()
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                embeding_idx = indices[i, j]
                if embeding_idx != -1:
                    passage_id = self.embedding_map[embeding_idx]
                    candidate_passage_scores[passage_id] += 1
                    retrieved_passage_ids.add(passage_id)

        candidate_gen_time = time.time()
        logging.info(f"Candidate generation {len(retrieved_passage_ids)} passages in {candidate_gen_time - faiss_search_time:.4f}s")

        if not retrieved_passage_ids:
            logging.warning("No candidate passages found.")
            return []
        
        # 3. re-ranking (ColBERT MaxSim Score - stage 2)
        final_scores = {}
        query_embeddings_tensor = torch.tensor(query_embeddings).to(self.encoder.device)

        candidate_passages_for_rerank = list(retrieved_passage_ids)

        logging.info(f"Re-rank {len(candidate_passages_for_rerank)} passages...")
        for passage_id in tqdm(candidate_passages_for_rerank):
            if passage_id not in self.passage_embedding_indices:
                logging.warning(f"Passage ID {passage_id} not found in passage_embedding_indices.")
                continue

            start_idx, end_idx = self.passage_embedding_indices[passage_id]
            doc_embeddings_tensor = self.all_doc_embeddings[start_idx:end_idx]
            if doc_embeddings_tensor.nelement() == 0:
                continue

            similarity_matrix = torch.einsum('qd,td->qt', query_embeddings_tensor, doc_embeddings_tensor)

            max_sim_scores, _ = torch.max(similarity_matrix, dim=1) # (query_maxlen,)

            final_passage_score = torch.sum(max_sim_scores).item()
            final_scores[passage_id] = final_passage_score

        rerank_time = time.time()
        logging.info(f"Re-ranking in {rerank_time - candidate_gen_time:.4f}s")

        # 4. Sort and return results
        sorted_passages = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)
        
        results = []
        for i in range(min(top_n, len(sorted_passages))):
            passage_id, score = sorted_passages[i]
            results.append({
                'passage_id': passage_id,
                'text': self.passage_map.get(passage_id, "Text not gound"),
                "score": score
            })
        
        total_time = time.time() - start_time
        logging.info(f"Total retrieval time: {total_time:.4f}s")
        return results
            