import streamlit as st
import requests
import json
import pandas as pd
from typing import List, Dict, Any
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from functools import wraps

# Set page config at the very beginning before any other Streamlit commands
st.set_page_config(
    page_title="Vector Search",
    page_icon="🔍",
    layout="wide"
)

# Constants
BACKEND_URL = "https://rude-doodles-drop.loca.lt"  # Your local tunnel URL
TIMEOUT = 10  # Timeout in seconds
MAX_RETRIES = 10  # Maximum number of retries for 502 errors

def create_retry_session():
    """Create a requests session with retry strategy"""
    session = requests.Session()
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=0.5,
        status_forcelist=[502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def with_retry(func):
    """Decorator to add retry logic to requests"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        session = create_retry_session()
        attempt = 0
        last_exception = None
        
        while attempt < MAX_RETRIES:
            try:
                kwargs['session'] = session
                return func(*args, **kwargs)
            except requests.exceptions.RequestException as e:
                attempt += 1
                last_exception = e
                if attempt < MAX_RETRIES:
                    time.sleep(0.5 * attempt)  # Exponential backoff
                    st.warning(f"Retry attempt {attempt}/{MAX_RETRIES}")
                continue
            break
        
        raise last_exception
    return wrapper

class VectorSearchUI:
    def __init__(self):
        self.initialize_session_state()
        self.check_backend_connection()

    @with_retry
    def check_backend_connection(self, session=None):
        try:
            with st.spinner("Checking backend connection..."):
                response = session.get(
                    f"{BACKEND_URL}/docs",
                    timeout=TIMEOUT,
                    headers={"Connection": "close"}
                )
                if response.status_code == 200:
                    st.sidebar.success("✅ Backend connected")
                else:
                    st.sidebar.error(f"❌ Backend connection failed with status code: {response.status_code}")
                    st.sidebar.info("Check if the backend API is responding correctly")
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

    @with_retry
    def update_configuration(self, config_data, session=None):
        response = session.post(
            f"{BACKEND_URL}/vector-search/configure",
            params=config_data,
            timeout=TIMEOUT,
            headers={"Connection": "close"}
        )
        return response

    def render_configuration_panel(self):
        st.sidebar.header("Vector Search Configuration")

        with st.sidebar.expander("Current Backend URL", expanded=False):
            st.code(BACKEND_URL)

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

        mixed_percentage = None
        if retrieval_weight == "Mixed":
            mixed_percentage = st.sidebar.slider(
                "Mixed Percentage",
                min_value=0,
                max_value=100,
                value=st.session_state.search_config["mixed_percentage"]
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
                "mixed_percentage": mixed_percentage if mixed_percentage is not None else 50,
                "rerank_enabled": rerank_enabled
            }
            
            with st.spinner("Updating configuration..."):
                try:
                    response = self.update_configuration(config_data)
                    if response.status_code == 200:
                        st.session_state.search_config = config_data
                        st.sidebar.success("✅ Configuration updated successfully!")
                    else:
                        st.sidebar.error(f"❌ Error updating configuration: {response.text}")
                except Exception as e:
                    st.sidebar.error(f"❌ Error: {str(e)}")

    @with_retry
    def perform_search(self, query, session=None):
        # First submit the query
        submit_response = session.post(
            f"{BACKEND_URL}/query/submit",
            json={"query": query},
            timeout=30
        )
        if submit_response.status_code != 200:
            raise Exception(f"Query submission failed: {submit_response.text}")

        # Then perform the retrieval
        retrieve_response = session.post(
            f"{BACKEND_URL}/query/retrieve",
            timeout=30
        )
        if retrieve_response.status_code != 200:
            raise Exception(f"Query retrieval failed: {retrieve_response.text}")

        return retrieve_response.json()

    @with_retry
    def perform_rerank(self, session=None):
        response = session.post(
            f"{BACKEND_URL}/vector-search/rerank",
            timeout=TIMEOUT,
            headers={"Connection": "close"}
        )
        return response

    @with_retry
    def get_results(self, session=None):
        response = session.get(
            f"{BACKEND_URL}/vector-search/results",
            timeout=TIMEOUT,
            headers={"Connection": "close"}
        )
        return response

    def render_search_interface(self):
        st.title("Vector Search Interface")
        st.subheader("Search")
        
        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_input("Enter your search query:", key="search_query")
        with col2:
            search_button = st.button("🔍 Search")

        if search_button and query:
            with st.spinner("Searching..."):
                try:
                    # Perform the search with automatic retries
                    search_results = self.perform_search(query)
                    
                    # If rerank is enabled, call the rerank endpoint
                    if st.session_state.search_config["rerank_enabled"]:
                        with st.spinner("Reranking results..."):
                            rerank_response = self.perform_rerank()
                            if rerank_response.status_code != 200:
                                st.warning("⚠️ Reranking failed, showing original results.")
                    
                    # Get the final results
                    results_response = self.get_results()
                    if results_response.status_code == 200:
                        st.session_state.search_results = results_response.json()["results"]
                        st.success(f"✨ Found {len(st.session_state.search_results)} results")
                    else:
                        st.error("❌ Failed to fetch search results")
                
                except Exception as e:
                    st.error(f"❌ Error performing search: {str(e)}")

    def render_results(self):
        if st.session_state.search_results:
            st.subheader("Search Results")
            
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
