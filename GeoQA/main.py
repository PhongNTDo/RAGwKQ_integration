import yaml
from src.encoding import Encoder
from src.indexing import FaissIndexer
from src.retreival import GeoRetriever
from src.utils import load_json, load_numpy, load_pickle, logging, load_memmap


def main():
    logging.info("Initializing retreiver...")
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    model_config = config['model']
    idx_config = config['indexing']
    ret_config = config['retrieval']
    general_config = config['general']
    
    # Load encoder
    encoder = Encoder(checkpoint=model_config['checkpoint'], 
                      device=general_config['device'],
                      doc_maxlen=model_config['doc_maxlen'],
                      query_maxlen=model_config['query_maxlen'])

    # Load indexer
    indexer = FaissIndexer(dim=model_config['embedding_dim'],
                           nlist=idx_config['faiss_nlist'],
                           m=idx_config['faiss_m'],
                           nbits=idx_config['faiss_nbits'])
    indexer.load(index_path=idx_config['index_save_path'], map_path=idx_config['embedding_map_save_path'])
    indexer.set_nprobe(ret_config['faiss_nprobe'])

    # Load necessary data for retriever
    passage_map = load_json(idx_config['passage_map_save_path'])
    all_doc_embeddings = load_memmap(idx_config['embedding_save_path'], shape=(len(passage_map), model_config['embedding_dim']))
    embedding_map = load_pickle(idx_config['embedding_map_save_path'])

    retriever = GeoRetriever(encoder=encoder,
                             indexer=indexer,
                             passage_map=passage_map,
                             all_doc_embeddings=all_doc_embeddings,
                             embedding_map=embedding_map)
    
    logging.info("Retriever ready. Enter queries (or type 'exit' to quit):")
    while True:
        query = input("Query: ")
        if query.lower() == 'exit':
            break
        if not query.strip():
            continue
        
        results = retriever.retrieve(query, candidate_k=ret_config['candidate_k'], top_n=ret_config['top_n'])
        if results:
            for i, res in enumerate(results):
                logging.info(f"Result {i+1} and Score: {res['score']:.4f}")
                logging.info(f"Passage {res['passage_id']}: {res['text']}")
        else:
            logging.info("No relevant passages found.")        


if __name__ == "__main__":
    main()