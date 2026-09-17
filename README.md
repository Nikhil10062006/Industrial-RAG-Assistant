#  Industrial RAG Assistant

### *The knowledge that walks out the door every time an expert retires — captured, structured, and made queryable, forever.*

<p align="center">
  <img src="https://img.shields.io/badge/RAG-Retrieval%20Augmented%20Generation-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/LLM-Mistral%20Nemotron-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Embeddings-NVIDIA%20NIM-76B900?style=for-the-badge&logo=nvidia" />
  <img src="https://img.shields.io/badge/Vector%20DB-ChromaDB-yellow?style=for-the-badge" />
  <img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit" />
</p>

<p align="center">
  <b>Built during the Tata Prashikshan Agentic AI Internship (Tata Steel)</b><br/>
  A working prototype for one of manufacturing's oldest, quietest problems.
</p>

<p align="center">
  <a href="https://industrial-rag-assistant.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀%20LIVE%20DEMO-Try%20it%20now-success?style=for-the-badge&logo=streamlit&logoColor=white" />
  </a>
</p>

<p align="center">
  <b>👉 <a href="https://industrial-rag-assistant.streamlit.app/">industrial-rag-assistant.streamlit.app</a> 👈</b>
</p>

---

##  The Problem Nobody Talks About

Every year, a 25-year veteran retires from a steel plant with **decades of undocumented fixes** locked in their head — the exact way to nurse a stubborn valve back to life, the sound a bearing makes right before it fails, the "trick" that isn't in any manual.

That knowledge doesn't get exported. It doesn't get backed up. **It just leaves.**

Meanwhile, a fresh technician on his first month is left guessing — afraid to keep bothering seniors with "small doubts," burning time waiting for someone to be free, repeating mistakes that were already solved a hundred times over.

> *"I've fixed this same problem a hundred times. I just wish someone had written it down properly."*
> — every retiring expert, everywhere

---

##  What This Actually Does

This isn't a chatbot bolted onto some PDFs. It's a **two-way knowledge bridge**:

| 👷 For the Junior Technician | 🧓 For the Senior Expert |
|---|---|
| Ask questions in plain language, 24/7 | Describe a fix the way you'd tell a colleague — no forms |
| Get step-by-step, source-cited answers instantly | The AI structures your words into a searchable entry |
| No fear of "looking incompetent" | Your expertise outlives your shift, your role, your retirement |

Under the hood, it's a full **Retrieval-Augmented Generation pipeline**:

```
Question → Embed → Retrieve relevant chunks → LLM generates a cited answer
New fix  → Structure via LLM → Embed → Add to knowledge base → Instantly searchable
```

---

##  Why This Isn't "Just Another RAG Wrapper"

✅ **Role-aware intelligence** : the same question gets a different answer depending on who's asking. A junior gets safety-first, step-by-step guidance. A senior gets terse, technical, edge-case-focused analysis. Same brain, different depth.

✅ **Live knowledge capture** : this isn't a static, one-time-ingested corpus. Experts can add new fixes *during natural conversation*, and the system re-embeds and makes them retrievable in real time.

✅ **Source-cited, not hallucinated** : every answer points back to exactly which document it came from, so nobody blindly trusts a black box on a shop floor.

✅ **Built to survive real-world API flakiness** : batched embeddings, exponential backoff retries, and graceful degradation baked in from day one, because production systems don't get to crash mid-demo.

---

##  Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
│   docs/     │─────▶│  Embedding    │─────▶│   ChromaDB       │
│ SOPs, fixes,│      │ (NVIDIA NIM)  │      │  Vector Store    │
│ safety docs │      └──────────────┘      └─────────────────┘
└─────────────┘                                     │
                                                     ▼
┌──────────────┐      ┌───────────────┐      ┌──────────────┐
│  Streamlit   │◀────▶│  RAG Pipeline  │◀────▶│ Mistral       │
│  Web UI      │      │  (Retrieval +  │      │ Nemotron LLM  │
│              │      │  Generation)   │      │               │
└──────────────┘      └───────────────┘      └──────────────┘
```

---

##  Quick Start

```bash
# Clone it
git clone https://github.com/Nikhil10062006/Industrial-RAG-Assistant.git
cd Industrial-RAG-Assistant

# Set up your environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt

# Add your NVIDIA API key
echo NVIDIA_API_KEY=your_key_here > .env

# Build the knowledge base (run the notebook once)
jupyter notebook notebook/rag_pipeline.ipynb

# Launch the assistant
streamlit run app.py
```

Open `http://localhost:8501` and start asking questions.

> 💡 **Prefer not to set anything up?** Just try the url https://industrial-rag-assistant.streamlit.app/  directly.

---

## 📂 What's Inside

```
├── app.py                  # Streamlit web app — the face of the assistant
├── rag_core.py              # Core pipeline: embeddings, vector store, retrieval, personalization, capture flow
├── notebook/
│   └── rag_pipeline.ipynb   # Full build process — ingestion, testing, stress-testing
├── docs/                    # Knowledge base source documents (SOPs, fixes, safety procedures)
└── requirements.txt
```

---


<p align="center">
  <i>Built by Nikhil Agarwal</i>
</p>
