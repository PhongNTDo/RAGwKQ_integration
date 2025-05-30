import os
import re
from tqdm import tqdm
from src.utils import logging


def basic_text_splitter(text, max_words, overlap_words):
    words  = re.findall(f'\S+', text)
    if not words:
        return []
    
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)

        if end == len(words):
            break
        start += max_words - overlap_words
        if start >= len(words):
            break
    return chunks


def process_wikipedia_directory(input_dir, max_words, overlap_words):
    passages = []
    passage_id_counter = 0
    if not os.path.isdir(input_dir):
        logging.error(f"Input directory '{input_dir}' does not exist.")
        return []
    
    for filename in tqdm(os.listdir(input_dir), desc="Processing files"):
        filepath = os.path.join(input_dir, filename)
        if not os.path.isfile(filepath):
            continue
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
            text = re.sub(r'\s+', ' ', text).strip()
            article_title = filename

            chunks = basic_text_splitter(text, max_words, overlap_words)
            for i, chunk in enumerate(chunks):
                if len(chunk.split()) > 10:
                    passages.append({
                        'passage_id': f"{article_title}_{passage_id_counter}",
                        'text': chunk,
                        'article_title': article_title
                    })
                    passage_id_counter += 1

    logging.info(f"Processed {len(passages)} passages.")
    return passages


def create_passages_map(passages):
    return {p['passage_id']: p['text'] for p in passages}
