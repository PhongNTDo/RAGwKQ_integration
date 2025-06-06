import json
import pickle
import logging
import numpy as np
import os


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_jsonl(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(json.dumps(line) + '\n')
    logging.info(f"Saved {len(data)} lines to {filepath}")


def load_jsonl(filepath):
    data = []
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        logging.info(f"Loaded {len(data)} lines from {filepath}")
    else:
        logging.warning(f"File {filepath} does not exist.")
    return data


def save_pickle(obj, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    logging.info(f"Saved object to {filepath}")


def load_pickle(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        logging.info(f"Loaded object from {filepath}")
        return obj
    else:
        logging.warning(f"File {filepath} does not exist.")
        return None
    

def save_numpy(array, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, array)
    logging.info(f"Saved numpy array with shape {array.shape} to {filepath}")


def load_numpy(filepath):
    if os.path.exists(filepath):
        array = np.load(filepath, allow_pickle=True)
        logging.info(f"Loaded numpy array with shape {array.shape} from {filepath}")
        return array
    else:
        logging.warning(f"File {filepath} does not exist.")
        return None
    

def save_json(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    logging.info(f"Saved data with {len(data)} items to {filepath}")


def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Loaded data with {len(data)} items from {filepath}")
        return data
    else:
        logging.warning(f"File {filepath} does not exist.")
        return None


def load_memmap(filepath, shape):
    if os.path.exists(filepath):
        array = np.memmap(filepath, dtype='float32', mode='r', shape=shape)
        logging.info(f"Loaded memmap with shape {array.shape} from {filepath}")
        return array
    else:
        logging.warning(f"File {filepath} does not exist.")
        return None
    
