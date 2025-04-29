import yaml
import numpy as np
from src.encoding import ColBERTEncoder
from src.indexing import FaissIndexer
from src.utils import load_jsonl, logging, load_numpy


if __name__ == "__main__":
    logging.info("Starting index building process")
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_config = config["model"]
    data_config = config["data"]
    idx_config = config["indexing"]
    general_config = config["general"]

    passages = load_jsonl(data_config["passages_path"])
    if not passages:
        logging.error("No passages found. Run preprocessing script first.")
        exit()

    encoder = ColBERTEncoder(
        checkpoint=model_config["checkpoint"],
        query_maxlen=model_config["query_maxlen"],
        doc_maxlen=model_config["doc_maxlen"],
        device=general_config["device"]
    )

    indexer = FaissIndexer(
        dim=model_config['embedding_dim'],
        nlist=idx_config['nlist'],
        m=idx_config['faiss_m'],
        nbits=idx_config['faiss_nbits']
    )

    logging.info("Encoding sample for Faiss training...")
    sample_size = min(idx_config['faiss_train_sample_size'], len(passages))
    sample_indices = np.random.choice(len(passages), sample_size, replace=False)
    sample_passages = [passages[i] for i in sample_indices]
    sample_texts = [p['text'] for p in sample_passages]

    training_embeddings_list = []
    for i in range(0, len(sample_texts), idx_config['batch_size']):
        batch_texts = sample_texts[i:i+idx_config['batch_size']]
        batch_embeddings = encoder.encode_passages_batch(batch_texts)
        for emb_array in batch_embeddings:
            if emb_array.shape[0] > 0:
                training_embeddings_list.append(emb_array)

    if not training_embeddings_list:
        logging.error("Could not generate any embeddings for training. Check encoder/data.")
        exit()

    training_vectors = np.concatenate(training_embeddings_list, axis=0)
    indexer.train_index(training_vectors)

    indexer.build_index(
        encoder=encoder,
        passages=passages,
        batch_size=idx_config['batch_size'],
        embeddings_save_path=idx_config['embeddings_save_path'],
        map_save_path=idx_config['map_save_path']
    )

    indexer.save_index(idx_config['index_save_path'])

    logging.info("Index building process finished.")