import streamlit as st
import requests
import json
import pandas as pd
from typing import List, Dict, Any
import time

# Set page config at the very beginning before any other Streamlit commands
st.set_page_config(
    page_title="Vector Search",
    page_icon="🔍",
    layout="wide"
)

# Constants
BACKEND_URL = "https://busy-kings-deny.loca.lt"  # Your local tunnel URL
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
                    headers={"Connection": "close"}  # Add this to prevent connection pooling issues
                )
                if response.status_code == 200:
                    st.sidebar.success("✅ Backend connected")
                else:
                    st.sidebar.error(f"❌ Backend connection failed with status code: {response.status_code}")
                    st.sidebar.info("Check if the backend API is responding correctly")
        except requests.exceptions.Timeout:
            st.sidebar.error("❌ Backend connection timed out")
            st.sidebar.info("The server is taking too long to respond. Check if it's running properly.")
        except requests.exceptions.ConnectionError:
            st.sidebar.error("❌ Cannot connect to backend")
            st.sidebar.info("Ensure the backend server is running and the URL is accessible")
        except Exception as e:
            st.sidebar.error(f"❌ Backend connection error: {str(e)}")
            st.sidebar.info("Unexpected error occurred while connecting to the backend")

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

        with st.sidebar.expander("Current Backend URL", expanded=False):
            st.code(BACKEND_URL)

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
            
            # Use the main area spinner instead of sidebar spinner
            with st.spinner("Updating configuration..."):
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
                except requests.exceptions.Timeout:
                    st.sidebar.error("❌ Request timed out. Please try again.")
                except requests.exceptions.ConnectionError:
                    st.sidebar.error("❌ Connection error. Please check if the backend is running.")
                except Exception as e:
                    st.sidebar.error(f"❌ Error: {str(e)}")

    def render_search_interface(self):
        st.title("Vector Search Interface")
        st.subheader("Search")
        
        # Search input and button
        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_input("Enter your search query:", key="search_query")
        with col2:
            search_button = st.button("🔍 Search")

        if search_button and query:
            with st.spinner("Searching..."):
                try:
                    # First, perform the vector search
                    response = requests.post(
                        f"{BACKEND_URL}/vector-search/retrieve",
                        params={"query": query},
                        timeout=30,  # Longer timeout for search
                        headers={"Connection": "close"}
                    )
                    
                    if response.status_code == 200:
                        # If rerank is enabled, call the rerank endpoint
                        if st.session_state.search_config["rerank_enabled"]:
                            with st.spinner("Reranking results..."):
                                rerank_response = requests.post(
                                    f"{BACKEND_URL}/vector-search/rerank",
                                    timeout=TIMEOUT,
                                    headers={"Connection": "close"}
                                )
                                if rerank_response.status_code != 200:
                                    st.warning("⚠️ Reranking failed, showing original results.")
                        
                        # Get the results
                        results_response = requests.get(
                            f"{BACKEND_URL}/vector-search/results",
                            timeout=TIMEOUT,
                            headers={"Connection": "close"}
                        )
                        if results_response.status_code == 200:
                            st.session_state.search_results = results_response.json()["results"]
                            st.success(f"✨ Found {len(st.session_state.search_results)} results")
                        else:
                            st.error("❌ Failed to fetch search results")
                    else:
                        st.error(f"❌ Search failed: {response.text}")
                
                except requests.exceptions.Timeout:
                    st.error("❌ Request timed out. Please try again.")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Connection error. Please check if the backend is running.")
                except Exception as e:
                    st.error(f"❌ Error performing search: {str(e)}")

    def render_results(self):
        if st.session_state.search_results:
            st.subheader("Search Results")
            
            # Add a download button for results
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
                    f"📄 Result {idx} - Correlation: {result['correlation']:.2f} - Tokens: {result['tokens']}"
                ):
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
    # Add CSS for better styling
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
