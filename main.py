import streamlit as st
import requests
import json
import pandas as pd
from typing import List, Dict, Any
import time

# Set page config at the very beginning
st.set_page_config(
    page_title="Vector Search",
    page_icon="🔍",
    layout="wide"
)

# Constants
BACKEND_URL = "https://honest-wolves-act.loca.lt"  # Your local tunnel URL
TIMEOUT = 10  # Timeout in seconds

class VectorSearchUI:
    def __init__(self):
        self.initialize_session_state()
        self.check_backend_connection()

    def check_backend_connection(self):
        try:
            with st.spinner("Checking backend connection..."):
                response = requests.get(
                    f"{BACKEND_URL}/docs",
                    timeout=TIMEOUT,
                    headers={"Connection": "close"}
                )
                if response.status_code == 200:
                    st.sidebar.success("✅ Backend connected")
                else:
                    st.sidebar.error(f"❌ Backend connection failed with status code: {response.status_code}")
        except Exception as e:
            st.sidebar.error(f"❌ Backend connection error: {str(e)}")

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
        if 'current_query' not in st.session_state:
            st.session_state.current_query = None

    def render_configuration_panel(self):
        st.sidebar.header("Vector Search Configuration")

        with st.sidebar.expander("Current Backend URL", expanded=False):
            st.code(BACKEND_URL)

        # Configuration inputs
        doc_correlation = st.sidebar.slider(
            "Doc Correlation",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.search_config["doc_correlation"],
            step=0.05
        )

        recall_number = st.sidebar.slider(
            "Recall Number",
            min_value=1,
            max_value=50,
            value=st.session_state.search_config["recall_number"]
        )

        retrieval_weight = st.sidebar.radio(
            "Knowledge Retrieval Weight",
            options=["Mixed", "Semantic", "Keyword"],
            index=["Mixed", "Semantic", "Keyword"].index(st.session_state.search_config["retrieval_weight"])
        )

        mixed_percentage = st.sidebar.slider(
            "Mixed Percentage",
            min_value=0,
            max_value=100,
            value=st.session_state.search_config["mixed_percentage"],
            help="Percentage balance between semantic and keyword search",
            disabled=(retrieval_weight != "Mixed")
        )

        rerank_enabled = st.sidebar.checkbox(
            "Enable Rerank Model",
            value=st.session_state.search_config["rerank_enabled"]
        )

        if st.sidebar.button("Apply Configuration"):
            config_data = {
                "doc_correlation": doc_correlation,
                "recall_number": recall_number,
                "retrieval_weight": retrieval_weight,
                "mixed_percentage": mixed_percentage if retrieval_weight == "Mixed" else None,
                "rerank_enabled": rerank_enabled
            }
            
            try:
                response = requests.post(
                    f"{BACKEND_URL}/vector-search/configure",
                    params=config_data,
                    timeout=TIMEOUT,
                    headers={"Connection": "close"}
                )
                if response.status_code == 200:
                    st.session_state.search_config = config_data
                    st.sidebar.success("✅ Configuration updated successfully!")
                else:
                    st.sidebar.error(f"❌ Error updating configuration: {response.text}")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {str(e)}")

    def render_search_interface(self):
        st.title("Vector Search Interface")
        st.subheader("Search")
        
        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_input("Enter your search query:", key="search_query")
            keywords = st.text_input("Enter keywords (optional, comma-separated):", key="search_keywords")
        with col2:
            search_button = st.button("🔍 Search")

        if search_button and query:
            keywords_list = [k.strip() for k in keywords.split(",")] if keywords else None
            self.perform_search(query, keywords_list)

    def perform_search(self, query: str, keywords: List[str] = None):
        with st.spinner("Searching..."):
            try:
                # Submit query
                query_data = {"query": query, "keywords": keywords}
                submit_response = requests.post(
                    f"{BACKEND_URL}/query/submit",
                    json=query_data,
                    timeout=TIMEOUT,
                    headers={"Connection": "close"}
                )
                
                if submit_response.status_code != 200:
                    st.error(f"❌ Failed to submit query: {submit_response.text}")
                    return

                # Retrieve results
                retrieve_response = requests.post(
                    f"{BACKEND_URL}/query/retrieve",
                    timeout=TIMEOUT,
                    headers={"Connection": "close"}
                )
                
                if retrieve_response.status_code == 200:
                    results = retrieve_response.json()
                    st.session_state.search_results = results.get("results", [])
                    st.success(f"✨ Found {len(st.session_state.search_results)} results")
                else:
                    st.error(f"❌ Failed to retrieve results: {retrieve_response.text}")

            except Exception as e:
                st.error(f"❌ Error performing search: {str(e)}")

    def render_results(self):
        if st.session_state.search_results:
            st.subheader("Search Results")
            
            # Download results button
            df = pd.DataFrame(st.session_state.search_results)
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name="search_results.csv",
                mime="text/csv",
            )
            
            for idx, result in enumerate(st.session_state.search_results, 1):
                with st.expander(
                    f"📄 Result {idx} - Correlation: {result.get('correlation', 0):.2f} - Tokens: {result.get('tokens', 0)}"
                ):
                    st.markdown(f"**Content:**\n{result.get('content', '')}")
                    if result.get('metadata'):
                        st.markdown("**Metadata:**")
                        for key, value in result['metadata'].items():
                            st.markdown(f"- {key}: {value}")

    def run(self):
        self.render_configuration_panel()
        self.render_search_interface()
        self.render_results()

def main():
    st.markdown("""
        <style>
        .stButton > button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
        }
        .stExpander {
            border: 1px solid #ddd;
            border-radius: 5px;
            margin-bottom: 1em;
        }
        </style>
    """, unsafe_allow_html=True)
    
    vector_search_ui = VectorSearchUI()
    vector_search_ui.run()

if __name__ == "__main__":
    main()
