import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import yaml
import numpy as np
from tqdm import tqdm
from src.encoding import Encoder
from src.indexing import FaissIndexer
from src.utils import load_jsonl, logging, load_numpy


if __name__ == "__main__":
    logging.info("Starting index building process")
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    
    model_config = config["model"]
    data_config = config["data"]
    index_config = config["indexing"]
    general_config = config["general"]

    passages = load_jsonl(data_config["processed_passage_path"])
    if not passages:
        logging.error("No passages found. Run preprocessing script first.")
        exit()

    encoder = Encoder(checkpoint=model_config['checkpoint'],
                      query_maxlen=model_config['query_maxlen'],
                      doc_maxlen=model_config['doc_maxlen'],
                      device=general_config['device'])
    
    indexer = FaissIndexer(dim=model_config['embedding_dim'],
                           nlist=index_config['faiss_nlist'],
                           m=index_config['faiss_m'],
                           nbits=index_config['faiss_nbits'])
    
    logging.info("Encoding sample for Faiss training...")
    sample_size = min(index_config['faiss_train_sample_size'], len(passages))
    sample_indices = np.random.choice(len(passages), sample_size, replace=False)
    sample_passages = [passages[i] for i in sample_indices]
    sample_texts = [p['text'] for p in sample_passages]

    training_embeddings_list = []
    for i in tqdm(range(0, len(sample_texts), index_config['batch_size'])):
        batch_texts = sample_texts[i:i+index_config['batch_size']]
        batch_embeddings = encoder.encode_passages_batch(batch_texts)
        for emb_array in batch_embeddings:
            if emb_array.shape[0] > 0:
                training_embeddings_list.append(emb_array)

    if not training_embeddings_list:
        logging.error("Could not generate any embeddings for training. Check encoder/data.")
        exit()

    # training_vectors = np.concatenate(training_embeddings_list, axis=0)
    training_vectors = np.array(training_embeddings_list)
    indexer.train_index(training_vectors)

    indexer.build_index(encoder=encoder,
                       passages=passages,
                       batch_size=index_config['batch_size'],
                       embeddings_save_path=index_config['embedding_save_path'],
                       map_save_path=index_config['embedding_map_save_path'],
                       idx_config=index_config)
    
    indexer.save(index_config['index_save_path'])

    logging.info("Index building process finished.")
