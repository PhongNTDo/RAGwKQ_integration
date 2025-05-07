import yaml
from src.encoding import ColBERTEncoder
from src.indexing import FaissIndexer
from src.retrieval import ColBERTRetriever
from src.utils import load_json, load_numpy, load_pickle, logging


if __name__ == "__main__":
    logging.info("Initializing retriever...")
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    model_config = config["model"]
    idx_config = config["index"]
    ret_config = config["retrieve"]
    general_config = config["general"]

    # Load encoder
    encoder = ColBERTEncoder(
        checkpoint=model_config["checkpoint"],
        query_maxlen=model_config["query_maxlen"],
        doc_maxlen=model_config["doc_maxlen"],
        device=model_config["device"],
    )

    # Load indexer
    indexer = FaissIndexer(
        dim=model_config["dim"],
        nlist=idx_config['faiss_nlist'],
        m=idx_config['faiss_m'],
        nbits=idx_config['faiss_nbits'],
    )
    indexer.load(index_path=idx_config["index_save_path"], map_path=idx_config['embedding_map_save_path'])
    indexer.set_nprobe(ret_config['faiss_nprobe'])

    # Load necessary data for retriever
    passage_map = load_json(idx_config['passage_map_save_path'])
    all_doc_embeddings = load_numpy(idx_config['embedding_save_path'])
    embedding_map = load_pickle(idx_config['embedding_map_save_path'])

    if passage_map is None or all_doc_embeddings is None or embedding_map is None:
        logging.error("Fail to load necessary data for retriever.")
        exit()

    # Initialize retriever
    retriever = ColBERTRetriever(
        encoder=encoder,
        indexer=indexer,
        passage_map=passage_map,
        all_doc_embeddings=all_doc_embeddings,
        embedding_map=embedding_map,
    )

    logging.info("Retriever ready. Enter queries (or type 'exit' to quit)")

    
