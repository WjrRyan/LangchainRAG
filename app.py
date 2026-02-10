"""
Streamlit Chat UI for the Agentic RAG system.

Run with:
    streamlit run app.py
"""

import uuid
import tempfile
from pathlib import Path

import streamlit as st

from config import DATA_DIR
from core.document_loader import SUPPORTED_EXTENSIONS
from core.text_splitter import split_documents
from core.document_loader import load_document
from core.vectorstore import VectorStoreManager
from agent.graph import create_app


# ─── Page Configuration ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="Agentic RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .step-box {
        background-color: #f0f2f6;
        border-left: 4px solid #4A90D9;
        padding: 10px 15px;
        margin: 5px 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.85em;
    }
    .citation-box {
        background-color: #e8f4e8;
        border-left: 4px solid #28a745;
        padding: 8px 12px;
        margin: 3px 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.82em;
    }
    .dark .step-box {
        background-color: #1e2130;
    }
    .dark .citation-box {
        background-color: #1a2e1a;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ────────────────────────────────────────────

def init_session_state():
    """Initialize all session state variables."""
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "vs_manager" not in st.session_state:
        st.session_state.vs_manager = VectorStoreManager()

    if "agent_app" not in st.session_state:
        app, config = create_app(st.session_state.thread_id)
        st.session_state.agent_app = app
        st.session_state.agent_config = config

    if "last_steps" not in st.session_state:
        st.session_state.last_steps = []

    if "last_citations" not in st.session_state:
        st.session_state.last_citations = []


init_session_state()


# ─── Sidebar: Document Management ───────────────────────────────────────────

with st.sidebar:
    st.header("📄 Knowledge Base")

    # Document upload
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "md", "csv"],
        accept_multiple_files=True,
        help="Supported formats: PDF, Markdown, CSV",
    )

    if uploaded_files:
        if st.button("📥 Ingest Documents", type="primary", use_container_width=True):
            with st.spinner("Processing documents..."):
                total_chunks = 0
                for uploaded_file in uploaded_files:
                    # Save to temp file
                    suffix = Path(uploaded_file.name).suffix
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix, dir=str(DATA_DIR)
                    ) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    try:
                        docs = load_document(tmp_path)
                        chunks = split_documents(docs)

                        # Update source metadata to show original filename
                        for chunk in chunks:
                            chunk.metadata["source"] = uploaded_file.name

                        st.session_state.vs_manager.add_documents(chunks)
                        total_chunks += len(chunks)
                        st.success(f"✅ {uploaded_file.name}: {len(chunks)} chunks")
                    except Exception as e:
                        st.error(f"❌ {uploaded_file.name}: {str(e)}")

                if total_chunks > 0:
                    st.success(f"Added {total_chunks} chunks to knowledge base!")

    # Knowledge base stats
    st.divider()
    doc_count = st.session_state.vs_manager.get_document_count()
    st.metric("Chunks in Knowledge Base", doc_count)

    # Clear knowledge base
    if doc_count > 0:
        if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
            st.session_state.vs_manager.clear()
            st.rerun()

    # New conversation
    st.divider()
    st.subheader("💬 Conversation")
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.last_steps = []
        st.session_state.last_citations = []
        app, config = create_app(st.session_state.thread_id)
        st.session_state.agent_app = app
        st.session_state.agent_config = config
        st.rerun()

    # Show agent reasoning toggle
    st.divider()
    st.session_state.show_reasoning = st.toggle(
        "Show Agent Reasoning",
        value=True,
        help="Display the agent's step-by-step thinking process",
    )


# ─── Main Chat Interface ────────────────────────────────────────────────────

st.title("🔍 Agentic RAG Assistant")
st.caption(
    "Upload documents to the knowledge base, then ask questions. "
    "The agent will route, retrieve, evaluate, and answer with citations."
)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show reasoning steps if available
        if msg["role"] == "assistant" and msg.get("steps") and st.session_state.get("show_reasoning"):
            with st.expander("🧠 Agent Reasoning Process", expanded=False):
                for step in msg["steps"]:
                    st.markdown(
                        f'<div class="step-box"><b>{step["step"]}</b>: {step["detail"]}</div>',
                        unsafe_allow_html=True,
                    )

        # Show citations if available
        if msg["role"] == "assistant" and msg.get("citations"):
            with st.expander(f"📚 Sources ({len(msg['citations'])})", expanded=False):
                for citation in msg["citations"]:
                    source = citation.get("source", "unknown")
                    parts = [f"📄 **{Path(source).name}**"]
                    if "page" in citation:
                        parts.append(f"Page {citation['page']}")
                    if "row" in citation:
                        parts.append(f"Row {citation['row']}")
                    if citation.get("type") == "web_search":
                        parts = [f"🌐 **{citation.get('title', source)}**"]
                        parts.append(f"[Link]({source})")
                    st.markdown(
                        f'<div class="citation-box">{" | ".join(parts)}</div>',
                        unsafe_allow_html=True,
                    )


# ─── Chat Input ──────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run the agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                initial_state = {
                    "question": prompt,
                    "original_question": prompt,
                    "chat_history": [],
                    "documents": [],
                    "generation": "",
                    "route": "",
                    "route_reasoning": "",
                    "query_rewrite_count": 0,
                    "web_search_needed": False,
                    "citations": [],
                    "sub_questions": [],
                    "sub_answers": [],
                    "steps": [],
                }

                result = st.session_state.agent_app.invoke(
                    initial_state,
                    config=st.session_state.agent_config,
                )

                answer = result.get("generation", "Sorry, I could not generate an answer.")
                steps = result.get("steps", [])
                citations = result.get("citations", [])

            except Exception as e:
                answer = f"An error occurred: {str(e)}"
                steps = []
                citations = []

        # Display answer
        st.markdown(answer)

        # Show reasoning steps
        if steps and st.session_state.get("show_reasoning"):
            with st.expander("🧠 Agent Reasoning Process", expanded=False):
                for step in steps:
                    st.markdown(
                        f'<div class="step-box"><b>{step["step"]}</b>: {step["detail"]}</div>',
                        unsafe_allow_html=True,
                    )

        # Show citations
        if citations:
            with st.expander(f"📚 Sources ({len(citations)})", expanded=False):
                for citation in citations:
                    source = citation.get("source", "unknown")
                    parts = [f"📄 **{Path(source).name}**"]
                    if "page" in citation:
                        parts.append(f"Page {citation['page']}")
                    if "row" in citation:
                        parts.append(f"Row {citation['row']}")
                    if citation.get("type") == "web_search":
                        parts = [f"🌐 **{citation.get('title', source)}**"]
                        parts.append(f"[Link]({source})")
                    st.markdown(
                        f'<div class="citation-box">{" | ".join(parts)}</div>',
                        unsafe_allow_html=True,
                    )

        # Store in session
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "steps": steps,
            "citations": citations,
        })
