# app.py

import streamlit as st
import nest_asyncio

from tools import (
    list_directory_contents,
    load_file_contents,
    perform_string_search,
    perform_semantic_search,
)
from load_repository import open_zipfile
from create_index import RepositoryParser
from agents import Agent, Runner
import yaml
import os

nest_asyncio.apply()

# ─── Page config & styling ─────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
      .block-container {
        padding: 4rem 5rem 0 5rem;
      }
      # header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# ─── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def index_repository(zip_file) -> None:
    """Unzip + index on first run; skip if same file/hash."""
    open_zipfile(zip_file)
    parser = RepositoryParser("repository/").split_documents()
    index = parser.index_documents(model="text-embedding-3-small")
    return index


def initialize_agent(config_path: str, model_id: str) -> Agent:
    """Load YAML + return a configured Agent."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return Agent(
        name=cfg["agent_name"],
        instructions=cfg["agent_instructions"],
        tools=[
            list_directory_contents,
            load_file_contents,
            perform_string_search,
            perform_semantic_search
        ],
        model=model_id
    )

# ─── Sidebar: credentials & repo upload ────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    # Secret management: prefer Streamlit secrets, fallback to env
    st.session_state.openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="••••••••••",
        key="api_key",
    )
    if st.session_state.openai_key:
        os.environ['OPENAI_API_KEY'] = st.session_state.openai_key

    AGENT_MODELS = {
        "gpt-4.1-mini": "gpt-4.1-mini",
        "gpt-o4-mini": "o4-mini",
        "gpt-4.1": "gpt-4.1"
    }
    model_choice = st.selectbox("Model", list(AGENT_MODELS.keys()))

    uploaded = st.file_uploader(
        "Repository ZIP", type="zip", help="Upload a .zip of your repo."
    )
    do_index = st.checkbox(
        "Index repository?", value=False,
        help="Index repository for semantic search. This can take several minutes for large repositories."
    )

    start = st.button("Start Session")
    if start:
        if not st.session_state.openai_key:
            st.warning("Please enter your OpenAI key.")
        elif not uploaded:
            st.warning("Please upload a ZIP file.")
        else:
            # instantiate & index
            st.session_state.github_agent = initialize_agent(
                "agent.yaml", AGENT_MODELS[model_choice]
            )
            if do_index:
                with st.spinner("Indexing repository..."):
                    st.session_state.index = index_repository(uploaded)
                st.success("Repository indexed!")
            st.session_state.ready = True

# ─── Main panel: chat interface ────────────────────────────────────────────────
# initialize welcome message
if "messages" not in st.session_state:
    welcome = "👋 Hi there! Upload a repository and add your API key to begin."
    st.session_state.messages = [{"role": "assistant", "content": welcome}]
    st.session_state.input_items = st.session_state.messages.copy()

# render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

ready = st.session_state.get("ready", False)
prompt = st.chat_input("Ask anything", disabled=not ready)

if prompt:
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.input_items.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = Runner.run_sync(
                        st.session_state.github_agent,
                        st.session_state.input_items,
                        max_turns=25
                )
                st.markdown(response.final_output)
            except Exception as e:
                st.error(f"Error: {e}")
                response = None

    if response:
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response.final_output})
        st.session_state.input_items = response.to_input_list()
