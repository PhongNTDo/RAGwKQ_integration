import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import streamlit as st
import yaml
import numpy as np
import json
import os
import pickle
from src.retreival import GeoRetriever
from src.indexing import FaissIndexer
from src.encoding import Encoder
from src.utils import load_json, load_numpy, load_pickle, logging, load_memmap

# --- Configuration Loading ---
CONFIG_PATH = "config/config.yaml"  # Assumes app.py is in GeoQA folder

@st.cache_data
def load_config():
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        st.error(f"Configuration file not found at {CONFIG_PATH}. Ensure it's in the GeoQA/config/ directory.")
        return None
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        return None

# --- Resource Loading (Models, Indexes, Maps) ---
@st.cache_resource
def load_embedding_model(model_name, device):
    try:
        model = SentenceTransformer(model_name, device=device)
        return model
    except Exception as e:
        st.error(f"Error loading embedding model '{model_name}': {e}")
        return None

@st.cache_resource
def load_faiss_index(index_path):
    try:
        # Ensure the path is correct relative to where the app is run (GeoQA folder)
        if not os.path.exists(index_path):
            st.error(f"FAISS index file not found at '{index_path}'. Please check the path in your config and file existence.")
            return None
        index = faiss.read_index(index_path)
        return index
    except Exception as e:
        st.error(f"Error loading FAISS index from '{index_path}': {e}")
        return None

@st.cache_data
def load_passage_map(passage_map_path):
    try:
        if not os.path.exists(passage_map_path):
            st.error(f"Passage map file not found at '{passage_map_path}'. Please check the path in your config and file existence.")
            return None
        with open(passage_map_path, 'r', encoding='utf-8') as f:
            passage_map = json.load(f)  # passage_id -> text
        return passage_map
    except Exception as e:
        st.error(f"Error loading passage map from '{passage_map_path}': {e}")
        return None

@st.cache_data
def load_embedding_to_passage_map(embedding_map_path):
    try:
        if not os.path.exists(embedding_map_path):
            st.error(f"Embedding-to-passage map file not found at '{embedding_map_path}'. Please check the path in your config and file existence.")
            return None
        with open(embedding_map_path, 'rb') as f:
            embedding_to_passage_map = pickle.load(f)  # embedding_idx -> passage_id
        return embedding_to_passage_map
    except Exception as e:
        st.error(f"Error loading embedding-to-passage map from '{embedding_map_path}': {e}")
        return None


@st.cache_resource
def load_full_retriever(config):
    """
    Loads all necessary components (models, indexes, data) and instantiates the main
    GeoRetriever object. This function is decorated with @st.cache_resource, so it
    will only run ONCE, and the returned 'retriever' object will be reused on every
    subsequent interaction.
    """
    st.info("Loading GeoQA resources... This will only happen once.")
    
    # Extract config sections
    model_config = config.get("model", {})
    idx_config = config.get("indexing", {})
    ret_config = config.get("retrieval", {})
    general_config = config.get("general", {})

    # Load necessary data files for the retriever
    try:
        passage_map = load_json(idx_config['passage_map_save_path'])
        embedding_shape = (len(passage_map), model_config['embedding_dim'])
        all_doc_embeddings = load_memmap(idx_config['embedding_save_path'], shape=embedding_shape)
        embedding_map = load_pickle(idx_config['embedding_map_save_path'])
    except Exception as e:
        st.error(f"Failed to load data files (maps, embeddings): {e}")
        return None

    # Load encoder model
    encoder = Encoder(
        checkpoint=model_config['checkpoint'],
        device=general_config['device'],
        doc_maxlen=model_config['doc_maxlen'],
        query_maxlen=model_config['query_maxlen']
    )

    # Load FAISS indexer
    indexer = FaissIndexer(
        dim=model_config['embedding_dim'],
        nlist=idx_config['faiss_nlist'],
        m=idx_config['faiss_m'],
        nbits=idx_config['faiss_nbits']
    )
    indexer.load(
        index_path=idx_config['index_save_path'],
        map_path=idx_config['embedding_map_save_path']
    )
    indexer.set_nprobe(ret_config['faiss_nprobe'])

    # Instantiate the final retriever object
    retriever = GeoRetriever(
        encoder=encoder,
        indexer=indexer,
        passage_map=passage_map,
        all_doc_embeddings=all_doc_embeddings,
        embedding_map=embedding_map
    )
    
    st.success("Resources loaded successfully! The app is ready.")
    return retriever


# --- Retrieval Logic ---
def retrieve_passages(query, retriever, candidate_k):
    if not query or not retriever:
        return []

    retrieved_passages_info = []
    results = retriever.retrieve(query, candidate_k=candidate_k)
    if results:
        for i, res in enumerate(results):
            # logging.info(f"Result {i+1} and Score: {res['score']:.4f}")
            # logging.info(f"Passage {res['passage_id']}: {res['text']}")

            retrieved_passages_info.append({"id": res['passage_id'],
                                           "text": res['text'],
                                           "score": round(res['score'], 3)})
    else:
        logging.info("No relevant passages found.")
        st.warning("No relevant passages found.")
        return []

    return retrieved_passages_info


# --- Main Streamlit App ---
def main():
    st.set_page_config(page_title="GeoQA Demo", layout="wide")
    st.title("🌍 GeoQA System Demo")

    config = load_config()
    if not config:
        st.stop()

    retriever = load_full_retriever(config)
    default_candidate_k = 3

    st.sidebar.header("⚙️ Query Configuration")
    query_text = st.sidebar.text_area("Enter your query:", height=100, placeholder="E.g., What is the capital of France?")
    
    candidate_k = st.sidebar.number_input(
        "Number of passages to retrieve (candidate_k):",
        min_value=1,
        max_value=128, # Adjust max as needed
        value=default_candidate_k,
        step=1
    )

    search_button = st.sidebar.button("🔍 Search", use_container_width=True)

    if search_button and query_text:
        st.subheader("📚 Search Results")
        with st.spinner("Retrieving relevant passages..."):
            retrieved_passages = retrieve_passages(
                query_text,
                retriever,
                candidate_k
            )

        if retrieved_passages:
            st.success(f"Found {len(retrieved_passages)} relevant passage(s).")
            
            for i, passage_info in enumerate(retrieved_passages):
                passage_text = passage_info["text"]
                passage_id = passage_info["id"]
                passage_score = passage_info["score"]
                
                with st.container(border=True):
                    # Display passage metadata
                    st.markdown(f"**Passage {i+1}** (ID: `{passage_id}`) | Score: **{passage_score}**")

                    # 1. Show the snippet of ~30 words
                    num_words_snippet = 30
                    words = passage_text.split()
                    if len(words) > num_words_snippet:
                        snippet = " ".join(words[:num_words_snippet]) + "..."
                    else:
                        snippet = passage_text  # Show full text if it's already short
                    
                    st.write(snippet) # Display the short snippet by default
                    st.write("---") # Add a small visual separator

                    col1, col2 = st.columns([0.7, 0.3]) # Give more space to the expander text

                    with col1:
                        with st.expander("👓 Review full passage"):
                            st.markdown(passage_text)

                    with col2:
                        download_filename = f"passage_{passage_id.replace('/', '_')}_{i+1}.txt"
                        st.download_button(
                            label="⬇️ Download",
                            data=passage_text.encode('utf-8'),
                            file_name=download_filename,
                            mime="text/plain",
                            key=f"download_btn_{passage_id}_{i}",
                            use_container_width=True # Make button fill the column
                        )
        else:
            st.info("No relevant passages found for your query. Try rephrasing or a different query.")
    elif search_button and not query_text:
        st.warning("Please enter a query to search.")

if __name__ == "__main__":
    main()
