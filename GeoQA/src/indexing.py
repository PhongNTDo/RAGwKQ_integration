import os
import faiss
import numpy as np
from tqdm import tqdm
from src.utils import logging


class FaissIndexer:
    def __init__(self, dim, nlist, m, nbits):
        self.dim = dim
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        
        quantizer = faiss.IndexFlatL2(self.dim)
        self.index = faiss.IndexIVFPQ(quantizer, self.dim, self.nlist, self.m, self.nbits)
        self.is_trained = self.index.is_trained
        self.embedding_id_to_passage_id = []

    def train_index(self, training_vectors):
        if training_vectors.shape[0] > self.nlist:
            logging.warning(f"Warning: Training data size {training_vectors.shape[0]} is smaller than nlist {self.nlist}. Reduce nlist or provide more data.")
        if not self.is_trained:
            logging.info(f"Training Faiss index with {training_vectors.shape[0]} vectors...")
            self.index.train(training_vectors)
            self.is_trained = True
            logging.info("Faiss index trained.")
        else:
            logging.info("Faiss index is already trained.")

    def build_index(self, encoder, passages, batch_size, embeddings_save_path, map_save_path, idx_config=None):
        if not self.is_trained:
            raise RuntimeError("Index must be trained before building")
        
        if idx_config is None:
            idx_config = {}
            logging.info("Starting pass 1/2: Counting total number of embeddings...")
            total_embeddings = 0
            for i in tqdm(range(0, len(passages), batch_size), desc="Counting embeddings"):
                batch_passages = passages[i:i+batch_size]
                batch_texts = [p['text'] for p in batch_passages]
                embs_list_for_count = encoder.encode_passages_batch(batch_texts)
                
                for passage_emb_array in embs_list_for_count:
                    if passage_emb_array.shape[0] > 0:
                        total_embeddings += passage_emb_array.shape[0]

            if total_embeddings == 0:
                logging.error("No embeddings were generated. Cannot build index.")
                return
            
            logging.info(f"Total number of embeddings: {total_embeddings}")
            
            logging.info(f"Initializing memap for embeddings at {embeddings_save_path}")
            os.makedirs(os.path.dirname(embeddings_save_path), exist_ok=True)
            all_doc_embeddings_mmap = np.memmap(embeddings_save_path, dtype='float32', mode='w+', shape=(total_embeddings, self.dim))

            logging.info(f"Encoding and adding vectors to Faiss indexer...")
            self.embedding_id_to_passage_id = []
            current_mmap_offset = 0

            FAISS_ADD_CHUNK_SIZE = idx_config['faiss_add_chunk_size_embedding']
            embeddings_for_current_faiss_chunk = []

            logging.info(f"Encoding {len(passages)} passages in batches of {batch_size} and building index...")
            for i in tqdm(range(0, len(passages), batch_size), desc="Encoding and adding vectors"):
                batch_passages = passages[i:i+batch_size]
                batch_texts = [p['text'] for p in batch_passages]
                embs_list_for