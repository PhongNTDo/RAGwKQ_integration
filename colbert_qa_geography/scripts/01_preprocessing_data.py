import yaml
from src.data_processing import process_wikipedia_directory, create_passage_map
from src.utils import save_jsonl, save_json, logging


if __name__ == "__main__":
    logging.info("Starting data preprocessing...")
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    data_config = config['data']

    passages = process_wikipedia_directory(
        input_dir=data_config['input_dir'],
        max_words=data_config['passage_max_words'],
        overlap_words=data_config['passage_overlap_words']
    )

    if passages:
        save_jsonl(passages, data_config['processed_passage_path'])
        passage_map = create_passage_map(passages)
        save_json(passage_map, config['indexing']['passage_map_save_path'])
        logging.info("Data preprocessing completed successfully.")
    else:
        logging.error("No passages were generated.")