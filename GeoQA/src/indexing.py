import os
import faiss
import numpy as np
from tqdm import tqdm
from src.utils import logging, save_pickle, load_pickle, save_numpy, load_numpy


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
                batch_ids = [p['passage_id'] for p in batch_passages]

                passage_embeddings_list = encoder.encode_passages_batch(batch_texts)

                for passage_idx, passage_emb_array in enumerate(passage_embeddings_list):
                    num_embeddings = passage_emb_array.shape[0]
                    if num_embeddings > 0:
                        passage_emb_array_f32 = passage_emb_array.astype('float32')
                        all_doc_embeddings_mmap[current_mmap_offset:current_mmap_offset+num_embeddings] = passage_emb_array_f32
                        current_mmap_offset += num_embeddings

                        embeddings_for_current_faiss_chunk.append(passage_emb_array_f32)
                        passage_id = batch_ids[passage_idx]
                        self.embedding_id_to_passage_id.extend([passage_id] * num_embeddings)

                        current_faiss_chunk_total_embeddings = sum([emb.shape[0] for emb in embeddings_for_current_faiss_chunk])
                        if current_faiss_chunk_total_embeddings >= FAISS_ADD_CHUNK_SIZE:
                            concatenated_faiss_chunk = np.concatenate(embeddings_for_current_faiss_chunk, axis=0)
                            self.index.add(concatenated_faiss_chunk)
                            embeddings_for_current_faiss_chunk = []
                            logging.info(f"Add {concatenated_faiss_chunk.shape[0]} vectors to Faiss index")

            if embeddings_for_current_faiss_chunk:
                concatenated_faiss_chunk = np.concatenate(embeddings_for_current_faiss_chunk, axis=0)
                self.index.add(concatenated_faiss_chunk)
                logging.info(f"Add final {concatenated_faiss_chunk.shape[0]} vectors to Faiss index")

            all_doc_embeddings_mmap.flush()
            del all_doc_embeddings_mmap

            logging.info(f"All embeddings saved to {embeddings_save_path}. Total: {current_mmap_offset} embeddings")
            logging.info("Faiss index building complete!")

            save_pickle(self.embedding_id_to_passage_id, map_save_path)

    def set_nprobe(self, nprobe):
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = nprobe
        else:
            logging.warning("CUrrent Faiss index type does not suppert nprobe")

    def search(self, query_embeddings, k):
        if not self.index.ntotal > 0:
            logging.warning("Index is empty")
            return np.array([]), np.array([])
        query_embeddings = np.array(query_embeddings).astype('float32')
        distances, indices = self.index.search(query_embeddings, k)
        return distances, indices

    def save(self, index_path):
        logging.info(f"Saving Faiss index to {index_path}")
        faiss.write_index(self.index, index_path)
        logging.info("Faiss index saved")

    def load(self, index_path, map_path):
        if os.path.exists(index_path):
            logging.info(f"Loading Faiss index from {index_path}")
            self.index = faiss.read_index(index_path)
            
            self.is_trained = self.index.is_trained
            logging.info(f"Faiss index loaded ({self.index.ntotal} vectors).")
        
            self.embedding_id_to_passage_id = load_pickle(map_path)
            if self.embedding_id_to_passage_id is None:
                logging.error("Failed to load embedding_id_to_passage_id.")
                self.embedding_id_to_passage_id = []
            else:
                logging.info(f"Loaded embedding_id_to_passage_id with {len(self.embedding_id_to_passage_id)} entries.")
        else:
            logging.error(f"Faiss index not found at {index_path}.")