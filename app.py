"""
app.py
Streamlit front end for the Tata Steel Agentic AI Knowledge Assistant.

Folder structure assumed:
    project_root/
        app.py          <- this file
        rag_core.py
        .env
        docs/
        data/           <- created by ChromaDB
        notebook/
            rag_pipeline.ipynb

Run from the project root with:
    streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints.chat_models import ChatNVIDIA

from rag_core import Embedding, VectorStore, RAGRetreiver, QueryRetreiver, CaptureFlow

load_dotenv()
nvidia_api_key = os.getenv("NVIDIA_API_KEY")

st.set_page_config(page_title="Tata Steel Knowledge Assistant", layout="wide")


# ---------------------------------------------------------------------------
# Cache heavy objects so they persist across interactions instead of
# reloading the embedding model / vector store on every click.
# ---------------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    embedding = Embedding()
    # app.py lives at the project root, so the vector store data folder
    # is "./data", not "../data" (which is correct only from notebook/).
    vector_store = VectorStore(persist_directory="./data")
    rag_retreiver = RAGRetreiver(vector_store, embedding)
    llm = ChatNVIDIA(
        model="mistralai/mistral-nemotron",
        nvidia_api_key=nvidia_api_key,
        timeout=120,
        max_retries=3
    )
    query_retriever = QueryRetreiver(rag_retreiver)
    capture_flow = CaptureFlow(vector_store, embedding, llm)
    return query_retriever, capture_flow, llm, vector_store


query_retriever, capture_flow, llm, vector_store = load_pipeline()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title(" Agentic AI Knowledge Assistant Tata Steel")
st.caption("Retrieve verified fixes & SOPs, or log new tacit knowledge from senior operators.")

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Select mode", [" Ask a question ", " Log a fix "])
    role = st.selectbox("Your role", ["junior", "senior"])
    st.divider()
    try:
        doc_count = vector_store.collection.count()
        st.caption(f"Knowledge base size: {doc_count} entries")
    except Exception:
        st.caption("Knowledge base size: unavailable")


if mode == " Ask a question ":
    st.subheader("Ask the assistant")
    query = st.text_input(
        "What do you want to know?",
        placeholder="e.g. Rolling mill high vibration — what should I check first?"
    )

    if st.button("Get Answer") and query.strip():
        with st.spinner("Searching knowledge base..."):
            try:
                answer, sources = query_retriever._rag_advanced(
                    query, role=role, llm=llm, return_context=True
                )
            except Exception as e:
                st.error(f"Something went wrong while generating the answer: {e}")
                answer, sources = None, []

        if answer:
            st.markdown("### Answer")
            st.write(answer)

            if sources:
                with st.expander(f"View sources ({len(sources)})"):
                    for s in sources:
                        st.markdown(
                            f"**{s['metadata'].get('source', 'unknown')}** "
                            f"— similarity: {s['similarity_score']:.2f}"
                        )
                        st.text(s['content'][:300] + ("..." if len(s['content']) > 300 else ""))

else:
    st.subheader("Log a new fix")
    st.caption("Describe a fix or workaround in plain language — the assistant will structure it and add it to the knowledge base.")

    raw_text = st.text_area(
        "Describe the fix in your own words",
        placeholder=(
            "e.g. pressure valve issue on Furnace 2, caused by a stuck relief "
            "valve after cold start, solved by cycling it twice before ignition"
        ),
        height=150
    )

    if st.button("Save to Knowledge Base") and raw_text.strip():
        with st.spinner("Structuring and saving..."):
            try:
                confirmation = capture_flow.capture(raw_text)
                st.success("Saved successfully")
                st.markdown(confirmation)
            except Exception as e:
                st.error(f"Something went wrong while saving: {e}")