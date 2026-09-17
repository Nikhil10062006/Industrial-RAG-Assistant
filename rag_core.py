"""
rag_core.py
Core RAG pipeline for the Tata Steel Agentic AI Knowledge Assistant.
Single source of truth — imported by both the notebook and app.py.
"""

import os
import re
import time
import uuid
from typing import List, Dict, Any

import numpy as np
import chromadb
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints.embeddings import NVIDIAEmbeddings

load_dotenv()
nvidia_api_key = os.getenv("NVIDIA_API_KEY")

def invoke_with_retry(llm, prompt, max_retries: int = 5, base_wait: int = 5):
    """
    Calls llm.invoke(prompt) with exponential backoff retry.
    NVIDIA's free-tier NIM endpoint occasionally returns transient 500s
    under load — this absorbs those instead of crashing the whole cell.
    """
    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            wait = base_wait * (2 ** attempt)
            print(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

class Embedding:
    def __init__(self, model_name: str = "nvidia/nemotron-3-embed-1b"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"Embeddings Model Name: {self.model_name}")
            self.model = NVIDIAEmbeddings(model=self.model_name, nvidia_api_key=nvidia_api_key)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Some error occurred while loading the embeddings model.\n{e}")
            raise

    def generate_embeddings(self, texts: List[str], batch_size: int = 10, max_retries: int = 3) -> np.ndarray:
        """Embeds texts in small batches with retry/backoff to avoid API timeouts on large sets."""
        if not self.model:
            raise ValueError("Model not loaded")
        if not texts:
            raise ValueError("No texts provided to embed")

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for attempt in range(max_retries):
                try:
                    batch_embeddings = self.model.embed_documents(batch)
                    all_embeddings.extend(batch_embeddings)
                    print(f"Embedded batch {i // batch_size + 1} ({len(batch)} docs)")
                    break
                except Exception as e:
                    wait = 2 ** attempt
                    print(f"Batch {i // batch_size + 1} failed (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        raise

        return np.array(all_embeddings)


class VectorStore:
    def __init__(self, collection_name: str = "knowledge_base", persist_directory: str = "../data"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "Description": "Document embeddings for the Knowledge Base of AAI_Learning_With_Development",
                    "hnsw:space": "cosine"
                }
            )
        except Exception as e:
            print(f"{e}")
            raise

    def add_docs(self, documents: List[Any], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("No of documents is not equal to the no of embeddings")

        ids, metadatas, document_text, embeddings_list = [], [], [], []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4()}_{i}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            document_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        try:
            self.collection.add(
                ids=ids,
                metadatas=metadatas,
                documents=document_text,
                embeddings=embeddings_list
            )
            print(f"Added {len(ids)} documents to the vector store.")
        except Exception as e:
            print(f"{e}")
            raise

    def reset_collection(self):
        """Wipes and recreates the collection — use before a clean re-ingestion run."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "Description": "Document embeddings for the Knowledge Base of AAI_Learning_With_Development",
                "hnsw:space": "cosine"
            }
        )
        print("Collection reset.")

class RAGRetreiver:
    def __init__(self, vector_store: VectorStore, embedding: Embedding):
        self.vector_store = vector_store
        self.embedding = embedding

    def _retreive(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []

        query_embedding = self.embedding.generate_embeddings([query])[0]
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            retrieved_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(
                    zip(ids, documents, metadatas, distances)
                ):
                    similarity_score = 1 - distance
                    if similarity_score > score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'rank': i + 1,
                            'similarity_score': similarity_score,
                            'distance': distance
                        })

            return retrieved_docs

        except Exception as e:
            print(f"{e}")
            return []

ROLE_PROFILES = {
    "junior": {
        "top_k": 3,
        "instruction": (
            "You are explaining this to a junior technician with limited experience. "
            "Use simple, step-by-step language. Avoid jargon, or explain it briefly if used. "
            "Be explicit about safety precautions. Keep the answer actionable and short."
        )
    },
    "senior": {
        "top_k": 4,  # kept modest to avoid oversized-context 500s from the LLM endpoint
        "instruction": (
            "You are explaining this to a senior engineer/operator with deep domain experience. "
            "You can use technical terminology directly. Be concise, skip basic explanations, "
            "and focus on edge cases, root causes, or anything non-obvious."
        )
    }
}


class QueryRetreiver:
    def __init__(self, rag_retreiver: RAGRetreiver):
        self.query = None
        self.retreiver = rag_retreiver

    def _rag_advanced(self, query, role="junior", llm=None, min_score=0.2, return_context=False):
        self.query = query
        profile = ROLE_PROFILES.get(role, ROLE_PROFILES["junior"])
        top_k = profile["top_k"]

        results = self.retreiver._retreive(query, top_k=top_k, score_threshold=min_score)

        if not results:
            msg = "I couldn't find relevant information in the knowledge base to answer that."
            return (msg, []) if return_context else msg

        context_blocks = []
        for doc in results:
            source = doc["metadata"].get("source", "unknown")
            context_blocks.append(f"[Source: {source}]\n{doc['content']}")
        context = "\n\n".join(context_blocks)

        prompt = f"""{profile['instruction']}

Use the following Context to answer the question asked by the user correctly.
Cite the source file for each piece of information you use, using the [Source: ...] labels provided.
If the context does not contain enough information to answer, say so clearly instead of guessing.

Context:{context}
Question: {query}
Answer:"""

        response = invoke_with_retry(llm, prompt)

        if return_context:
            return response.content, results
        return response.content


class SimpleDoc:
    """Lightweight stand-in for a langchain Document, used for captured entries."""
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


class CaptureFlow:
    def __init__(self, vector_store: VectorStore, embedding: Embedding, llm):
        self.vector_store = vector_store
        self.embedding = embedding
        self.llm = llm

    def structure_entry(self, raw_text: str) -> dict:
        prompt = f"""A senior technician described a fix informally. Convert it into a structured knowledge entry.

Raw input: "{raw_text}"

Respond ONLY in this exact format (no extra text):
Title: <short title>
Category: <one of: furnace, rolling_mill, safety, electrical, general>
Problem: <what went wrong>
Solution: <how it was fixed>
Tags: <comma-separated keywords>"""

        response = invoke_with_retry(self.llm, prompt)
        text = response.content

        def extract(field):
            match = re.search(rf"{field}:\s*(.+)", text)
            return match.group(1).strip() if match else ""

        return {
            "title": extract("Title"),
            "category": extract("Category"),
            "problem": extract("Problem"),
            "solution": extract("Solution"),
            "tags": extract("Tags"),
            "raw_text": raw_text
        }

    def capture(self, raw_text: str):
        structured = self.structure_entry(raw_text)

        content = (
            f"Title: {structured['title']}\n"
            f"Problem: {structured['problem']}\n"
            f"Solution: {structured['solution']}\n"
            f"Tags: {structured['tags']}"
        )

        doc = SimpleDoc(
            page_content=content,
            metadata={
                "source": f"captured_{structured['title'][:30].replace(' ', '_')}.txt",
                "category": structured["category"] or "general",
                "captured": True
            }
        )

        embedding_vec = self.embedding.generate_embeddings([content])
        self.vector_store.add_docs([doc], embedding_vec)

        confirmation = (
            f"Got it — logged as:\n"
            f"Title: {structured['title']}\n"
            f"Category: {structured['category']}\n"
            f"Solution: {structured['solution']}"
        )
        return confirmation


def load_docs_with_category(base_dir: str = "../docs"):
    """Walks base_dir, loading every .txt file and tagging it with its
    subfolder name as 'category' metadata. Returns a list of Document objects."""
    from langchain_community.document_loaders import TextLoader

    documents = []
    for root, _, files in os.walk(base_dir):
        for fname in files:
            if fname.endswith(".txt"):
                path = os.path.join(root, fname)
                category = os.path.relpath(root, base_dir)
                loader = TextLoader(path, encoding="utf-8")
                docs = loader.load()
                for d in docs:
                    d.metadata["category"] = category if category != "." else "general"
                    d.metadata["source"] = fname
                documents.extend(docs)
    print(f"Loaded {len(documents)} documents across categories")
    return documents