import streamlit as st
import requests
import json
import pandas as pd
from typing import List, Dict, Any
import time

# Constants
BACKEND_URL = "https://busy-kings-deny.loca.lt"  # Update this if your backend is on a different URL

class VectorSearchUI:
    def __init__(self):
        self.setup_page()
        self.initialize_session_state()

    def setup_page(self):
        st.set_page_config(
            page_title="Vector Search Interface",
            page_icon="🔍",
            layout="wide"
        )
        st.title("Vector Search Interface")

    def initialize_session_state(self):
        if 'search_results' not in st.session_state:
            st.session_state.search_results = []
        if 'search_config' not in st.session_state:
            st.session_state.search_config = {
                "doc_correlation": 0.85,
                "recall_number": 10,
                "retrieval_weight": "Mixed",
                "mixed_percentage": 50,
                "rerank_enabled": False
            }

    def render_configuration_panel(self):
        st.sidebar.header("Vector Search Configuration")

        # Doc Correlation slider
        doc_correlation = st.sidebar.slider(
            "Doc Correlation",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.search_config["doc_correlation"],
            step=0.05,
            help="Set the minimum correlation threshold for document matching"
        )

        # Recall Number slider
        recall_number = st.sidebar.slider(
            "Recall Number",
            min_value=1,
            max_value=50,
            value=st.session_state.search_config["recall_number"],
            help="Number of documents to retrieve"
        )

        # Retrieval Weight radio
        retrieval_weight = st.sidebar.radio(
            "Knowledge Retrieval Weight",
            options=["Mixed", "Semantic", "Keyword"],
            index=["Mixed", "Semantic", "Keyword"].index(st.session_state.search_config["retrieval_weight"])
        )

        # Mixed Percentage slider (only show if Mixed is selected)
        mixed_percentage = None
        if retrieval_weight == "Mixed":
            mixed_percentage = st.sidebar.slider(
                "Mixed Percentage",
                min_value=0,
                max_value=100,
                value=st.session_state.search_config["mixed_percentage"],
                help="Percentage balance between semantic and keyword search"
            )

        # Rerank Model checkbox
        rerank_enabled = st.sidebar.checkbox(
            "Enable Rerank Model",
            value=st.session_state.search_config["rerank_enabled"]
        )

        # Apply Configuration button
        if st.sidebar.button("Apply Configuration"):
            config_data = {
                "doc_correlation": doc_correlation,
                "recall_number": recall_number,
                "retrieval_weight": retrieval_weight,
                "mixed_percentage": mixed_percentage if mixed_percentage is not None else 50,
                "rerank_enabled": rerank_enabled
            }
            
            try:
                response = requests.post(
                    f"{BACKEND_URL}/vector-search/configure",
                    params=config_data
                )
                if response.status_code == 200:
                    st.session_state.search_config = config_data
                    st.sidebar.success("Configuration updated successfully!")
                else:
                    st.sidebar.error(f"Error updating configuration: {response.text}")
            except Exception as e:
                st.sidebar.error(f"Error connecting to backend: {str(e)}")

    def render_search_interface(self):
        st.subheader("Search")
        
        # Search input and button
        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_input("Enter your search query:", key="search_query")
        with col2:
            search_button = st.button("Search")

        if search_button and query:
            try:
                # First, perform the vector search
                response = requests.post(
                    f"{BACKEND_URL}/vector-search/retrieve",
                    params={"query": query}
                )
                
                if response.status_code == 200:
                    # If rerank is enabled, call the rerank endpoint
                    if st.session_state.search_config["rerank_enabled"]:
                        rerank_response = requests.post(f"{BACKEND_URL}/vector-search/rerank")
                        if rerank_response.status_code != 200:
                            st.warning("Reranking failed, showing original results.")
                    
                    # Get the results
                    results_response = requests.get(f"{BACKEND_URL}/vector-search/results")
                    if results_response.status_code == 200:
                        st.session_state.search_results = results_response.json()["results"]
                        st.success(f"Found {len(st.session_state.search_results)} results")
                    else:
                        st.error("Failed to fetch search results")
                else:
                    st.error(f"Search failed: {response.text}")
            
            except Exception as e:
                st.error(f"Error performing search: {str(e)}")

    def render_results(self):
        if st.session_state.search_results:
            st.subheader("Search Results")
            
            for idx, result in enumerate(st.session_state.search_results, 1):
                with st.expander(f"Result {idx} - Correlation: {result['correlation']:.2f} - Tokens: {result['tokens']}"):
                    st.markdown(f"**Content:**\n{result['content']}")
                    if result.get('metadata'):
                        st.markdown("**Metadata:**")
                        for key, value in result['metadata'].items():
                            st.markdown(f"- {key}: {value}")

    def run(self):
        self.render_configuration_panel()
        self.render_search_interface()
        self.render_results()

def main():
    vector_search_ui = VectorSearchUI()
    vector_search_ui.run()

if __name__ == "__main__":
    main()
