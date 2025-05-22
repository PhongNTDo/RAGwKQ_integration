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

    def build_index(self, encoder, passages, batch_size, embeddings_save_path, map_save_path, idx_config=None):
        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding vectors.")
        
        # all_doc_embeddings_list = []
        if idx_config is None:
            idx_config = {}
        logging.info("Starting pass 1/2: Counting total number of embeddings...")
        total_embeddings_count = 0
        for i in tqdm(range(0, len(passages), batch_size), desc="Pass 1/2: Counting Embeddings"):
            batch_passages_data = passages[i:i+batch_size]
            batch_texts = [p['text'] for p in batch_passages_data]

            embs_list_for_count = encoder.encode_passages_batch(batch_texts)
            
            for passage_emb_array in embs_list_for_count:
                if passage_emb_array.shape[0] > 0:
                    total_embeddings_count += passage_emb_array.shape[0]

        if total_embeddings_count == 0:
            logging.error("No embeddings were generated. Cannot build index.")
            return

        logging.info(f"Total individual token embeddings to be processed: {total_embeddings_count}")

        logging.info(f"Initializing memap for embeddings at {embeddings_save_path} with shape ({total_embeddings_count},{self.dim})")
        os.makedirs(os.path.dirname(embeddings_save_path), exist_ok=True)
        all_doc_embeddings_mmap = np.memmap(embeddings_save_path, dtype='float32', mode='w+', shape=(total_embeddings_count, self.dim))

        logging.info("Starting pass 2/2: Encoding and adding vectors to Faiss index...")
        self.embedding_id_to_passage_id = []
        current_mmap_offset = 0

        FAISS_ADD_CHUNK_SIZE_EMBEDDINGS = idx_config['faiss_add_chunk_size_embeddings'] if 'faiss_add_chunk_size_embeddings' in idx_config else 100000
        embeddings_for_current_faiss_chunk = []

        logging.info(f"Encoding {len(passages)} passages in batches of {batch_size} and building index...")
        for i in range(0, len(passages), batch_size):
            batch_passages_data = passages[i:i+batch_size]
            batch_texts = [p['text'] for p in batch_passages_data]
            batch_ids = [p['passage_id'] for p in batch_passages_data]

            passage_embeddings_list = encoder.encode_passages_batch(batch_texts)

            for passage_idx, passage_emb_array in enumerate(passage_embeddings_list):
                num_embeddings_in_passage = passage_emb_array.shape[0]
                if num_embeddings_in_passage > 0:
                    passage_emb_array_f32 = passage_emb_array.astype(np.float32)
                    
                    all_doc_embeddings_mmap[current_mmap_offset:current_mmap_offset+num_embeddings_in_passage] = passage_emb_array_f32
                    
                    embeddings_for_current_faiss_chunk.append(passage_emb_array_f32)

                    passage_id = batch_ids[passage_idx]
                    self.embedding_id_to_passage_id.extend([passage_id] * num_embeddings_in_passage)
                    
                    current_mmap_offset += num_embeddings_in_passage

                    current_faiss_chunk_total_embeddings = sum(e.shape[0] for e in embeddings_for_current_faiss_chunk)
                    if current_faiss_chunk_total_embeddings >= FAISS_ADD_CHUNK_SIZE_EMBEDDINGS:
                        concatenated_faiss_chunk = np.concatenate(embeddings_for_current_faiss_chunk, axis=0)
                        self.index.add(concatenated_faiss_chunk)
                        embeddings_for_current_faiss_chunk = []
                        logging.info(f"Add {concatenated_faiss_chunk.shape[0]} embeddings to Faiss. Total in index: {self.index.ntotal}. And progress achieve: {i}/{len(passages) - 1}")

        if embeddings_for_current_faiss_chunk:
            concatenated_faiss_chunk = np.concatenate(embeddings_for_current_faiss_chunk, axis=0)
            self.index.add(concatenated_faiss_chunk)
            logging.info(f"Add final {concatenated_faiss_chunk.shape[0]} embeddings to Faiss. Total in index: {self.index.ntotal}")
        
        all_doc_embeddings_mmap.flush()
        del all_doc_embeddings_mmap
        logging.info(f"All embeddings saved to {embeddings_save_path}. Total: {current_mmap_offset}")
        
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