import streamlit as st
import yaml
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os
import pickle

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

# --- Retrieval Logic ---
def retrieve_passages(query, model, index, passage_map, embedding_to_passage_map, candidate_k, faiss_nprobe):
    if not query or not model or not index or not passage_map or not embedding_to_passage_map:
        return []

    if hasattr(index, 'nprobe'):
        index.nprobe = faiss_nprobe
    
    query_embedding = model.encode([query], normalize_embeddings=True)

    if query_embedding is None:
        st.error("Failed to embed query.")
        return []

    try:
        distances, embedding_indices = index.search(query_embedding, candidate_k)
    except Exception as e:
        st.error(f"Error during FAISS search: {e}")
        return []

    retrieved_passages_info = []
    if embedding_indices.size > 0:
        for i in range(embedding_indices.shape[1]):
            embedding_idx = embedding_indices[0, i]
            if embedding_idx == -1:  # FAISS can return -1
                continue

            # Try int key first for embedding_to_passage_map, then str as fallback
            passage_id_from_map = embedding_to_passage_map.get(int(embedding_idx))
            if passage_id_from_map is None:
                passage_id_from_map = embedding_to_passage_map.get(str(embedding_idx))

            if passage_id_from_map is not None:
                # passage_map (from JSON) keys are strings.
                passage_text = passage_map.get(str(passage_id_from_map))
                if passage_text:
                    retrieved_passages_info.append({
                        "id": str(passage_id_from_map), 
                        "text": passage_text, 
                        "score": float(distances[0, i])
                    })
                else:
                    st.warning(f"Passage text not found for passage_id: {passage_id_from_map} (from embedding_idx: {embedding_idx})")
            else:
                st.warning(f"Passage ID not found for embedding_idx: {embedding_idx}")

    return retrieved_passages_info

# --- Main Streamlit App ---
def main():
    st.set_page_config(page_title="GeoQA Demo", layout="wide")
    st.title("🌍 GeoQA System Demo")

    config = load_config()
    if not config:
        st.stop()

    model_conf = config.get("model", {})
    indexing_conf = config.get("indexing", {})
    retrieval_conf = config.get("retrieval", {})
    general_conf = config.get("general", {})

    model_checkpoint = model_conf.get("checkpoint", "BAAI/bge-m3")
    device = general_conf.get("device", "cpu")

    index_save_path = indexing_conf.get("index_save_path")
    passage_map_save_path = indexing_conf.get("passage_map_save_path")
    embedding_map_save_path = indexing_conf.get("embedding_map_save_path")

    default_candidate_k = retrieval_conf.get("candidate_k", 3)
    faiss_nprobe = retrieval_conf.get("faiss_nprobe", 16)

    # Load resources
    # Paths in config are relative to project root (GeoQA folder)
    model = load_embedding_model(model_checkpoint, device)
    faiss_index = load_faiss_index(index_save_path)
    passage_map = load_passage_map(passage_map_save_path)
    embedding_to_passage_map = load_embedding_to_passage_map(embedding_map_save_path)

    if not all([model, faiss_index, passage_map, embedding_to_passage_map]):
        st.error("Failed to load one or more essential resources. Please check paths in config.yaml and ensure files exist.")
        st.info(f"Expected paths:\n"
                f"- Index: {os.path.abspath(index_save_path) if index_save_path else 'Not configured'}\n"
                f"- Passage Map: {os.path.abspath(passage_map_save_path) if passage_map_save_path else 'Not configured'}\n"
                f"- Embedding Map: {os.path.abspath(embedding_map_save_path) if embedding_map_save_path else 'Not configured'}")
        st.stop()

    st.sidebar.header("⚙️ Query Configuration")
    query_text = st.sidebar.text_area("Enter your query:", height=100, placeholder="E.g., What is the capital of France?")
    
    candidate_k = st.sidebar.number_input(
        "Number of passages to retrieve (candidate_k):",
        min_value=1,
        max_value=50, # Adjust max as needed
        value=default_candidate_k,
        step=1
    )

    search_button = st.sidebar.button("🔍 Search", use_container_width=True)

    if search_button and query_text:
        st.subheader("📚 Search Results")
        with st.spinner("Retrieving relevant passages..."):
            retrieved_passages = retrieve_passages(
                query_text,
                model,
                faiss_index,
                passage_map,
                embedding_to_passage_map,
                candidate_k,
                faiss_nprobe
            )

        if retrieved_passages:
            st.success(f"Found {len(retrieved_passages)} relevant passage(s).")
            
            for i, passage_info in enumerate(retrieved_passages):
                passage_text = passage_info["text"]
                passage_id = passage_info["id"]
                
                container = st.container(border=True)
                container.markdown(f"**Passage {i+1} (ID: `{passage_id}`)**")

                num_words_snippet = 30  # Number of words for the snippet
                snippet = " ".join(passage_text.split()[:num_words_snippet]) + "..."
                container.write(snippet)

                download_filename = f"passage_{passage_id.replace('/', '_')}_{i+1}.txt" # Sanitize passage_id for filename
                container.download_button(
                    label=f"📥 Download full passage {i+1}",
                    data=passage_text.encode('utf-8'), # Encode to bytes
                    file_name=download_filename,
                    mime="text/plain",
                    key=f"download_btn_{passage_id}_{i}"
                )
        else:
            st.info("No relevant passages found for your query. Try rephrasing or a different query.")
    elif search_button and not query_text:
        st.warning("Please enter a query to search.")

if __name__ == "__main__":
    main()