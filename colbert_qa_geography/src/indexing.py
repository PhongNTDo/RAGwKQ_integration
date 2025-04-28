import os
import faiss
import numpy as np
from tqdm import tqdm
import pickle
from src.utils import logging, save_pickle, load_pickle, save_numpy, load_numpy


class FaissIndexer:
    def __init__(self, dim, nlist, m, nbits):
        self.dim = dim
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        
        quantizer = faiss.IndexFlatL2(self.dim)
        self.index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
        self.is_trained = self.index.is_trained
        self.embedding_id_to_passage_id = []
        logging.info(f"Initialized Faiss IndexIVFPQ (d={dim}, nlist={nlist}, m={m}, nbits={nbits})")

    def train_index(self, training_vectors):
        if training_vectors.shape[0] > self.nlist:
            logging.warning(f"Warning: Training data size {training_vectors.shape[0]} is smaller than nlist {self.nlist}. Reduce nlist or provide more data.")
        if not self.is_trained:
            logging.info(f"Training Faiss index on {training_vectors.shape[0]} vectors...")
            self.index.train(training_vectors)
            self.is_trained = True
            logging.info(f"Faiss index training complete.")
        else:
            logging.info(f"Faiss index is already trained.")

    def build_index(self, encoder, passages, batch_size, embeddings_save_path, map_save_path):
        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding vectors.")
        
        all_doc_embeddings_list = []
        self.embedding_id_to_passage_id = []
        global_embedding_idx = 0

        logging.info(f"Encoding {len(passages)} passages in batches of {batch_size}...")
        for i in tqdm(range(0, len(passages), batch_size), desc="Encoding Passages"):
            batch_passages = passages[i:i+batch_size]
            batch_texts = [p['text'] for p in batch_passages]
            batch_ids = [p['passage_id'] for p in batch_passages]
            
            passage_embeddings_list = encoder.encode_passage_batch(batch_texts)

            for passage_idx, passage_emb_array in enumerate(passage_embeddings_list):
                if passage_emb_array.shape[0] > 0:
                    all_doc_embeddings_list.append(passage_emb_array)
                    num_embeddings_in_passage = passage_emb_array.shape[0]
                    passage_id = batch_ids[passage_idx]
                    self.embedding_id_to_passage_id.extend([passage_id] * num_embeddings_in_passage)
                    global_embedding_idx += num_embeddings_in_passage

        if not all_doc_embeddings_list:
            logging.error("No embeddings were generated. Cannot build index.")
            return
        
        all_doc_embeddings = np.concatenate(all_doc_embeddings_list, axis=0).astype('float32')
        logging.info(f"Total individual token embeedings: {all_doc_embeddings.shape[0]}")
        save_numpy(all_doc_embeddings, embeddings_save_path)

        logging.info("Adding vectors to Faiss index...")
        self.index.add(all_doc_embeddings)
        logging.info("Faiss index building complete.")

        save_pickle(self.embedding_id_to_passage_id, map_save_path)

    def set_nprobe(self, nprobe):
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = nprobe
            logging.info(f"Set nprobe to {nprobe}")
        else:
            logging.warning("Current Faiss index type does not support nprobe setting")

    def search(self, query_embeddings, k):
        if not self.index.ntotal > 0:
            logging.warning("Faiss index is empty. Cannot perform search.")
            return np.array([]), np.array([])
        
        query_embeddings = query_embeddings.astype('float32')
        distances, indices = self.index.search(query_embeddings, k)
        return distances, indices
    
    def save(self, index_path):
        logging.info(f"Saving Faiss index to {index_path}...")
        faiss.write_index(self.index, index_path)
        logging.info("Faiss index saved.")

    def load(self, index_path, map_path):
        if os.path.exists(index_path):
            logging.info(f"Loading Faiss index from {index_path}...")
            self.index = faiss.read_index(index_path)
            
            self.is_trained = self.index.is_trained
            logging.info(f"Faiss index loaded ({self.index.ntotal} vectors).")

            self.embedding_id_to_passage_id = load_pickle(map_path)
            if self.embedding_id_to_passage_id is None:
                logging.error("Failed to load embedding_id_to_passage_id.")
                self.embedding_id_to_passage_id = []
            else:
                logging.info(f"Loaded embedding map for {len(self.embedding_id_to_passage_id)} embeddings.")
        else:
            logging.warning(f"Faiss index file {index_path} does not exist.")