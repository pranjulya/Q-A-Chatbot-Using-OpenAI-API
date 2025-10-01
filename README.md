# Q&A Chatbot Using OpenAI API

An end-to-end question-answering chatbot that grounds responses in your own documents, FAQs, or knowledge base. The project combines document parsing, semantic search, and OpenAI's GPT models to deliver accurate, source-backed answers through a lightweight web interface.

## Why This Project?
- **Personalized assistance:** Point the chatbot at course notes, company handbooks, or portfolio material to deliver tailored answers.
- **Rapid prototyping:** Build a retrieval-augmented generation (RAG) system without managing heavy infrastructure.
- **Learning opportunity:** Practice with Python, OpenAI APIs, embeddings, and simple web deployment.

## Core Features
- Document upload & ingestion (PDF, DOCX, Markdown, plain text)
- Text cleaning, chunking, and metadata tracking for citations
- Embedding-based semantic search powered by OpenAI `text-embedding-3` models
- GPT-4/GPT-3.5 answer synthesis with grounded context paragraphs
- Flask or Streamlit front end with conversational UI
- Optional answer streaming, citation display, and feedback capture

## Architecture Overview
1. **Document Ingestion**
   - Parse supported files with `pdfplumber`, `python-docx`, or Markdown readers.
   - Normalize text (remove headers/footers, trim whitespace) and chunk into ~500 token windows.
   - Store chunks with document metadata (title, page, section).
2. **Vector Store & Retrieval**
   - Generate embeddings via OpenAI's `text-embedding-3-small` (cost-effective) or `-large` (higher recall).
   - Persist vectors in FAISS, Chroma, or Supabase pgvector; track chunk metadata for citations.
   - On each query, embed the user question and retrieve the top-k similar chunks.
3. **Answer Generation**
   - Build a system prompt stressing grounded answers and citation formatting.
   - Call GPT-4 Turbo (primary) or GPT-3.5 Turbo (fallback) with retrieved context.
   - Optionally include re-ranking, refusal logic, and answer validation.
4. **User Interface**
   - Present a chat panel with conversation history, streaming responses, and cited sources.
   - Provide document management (upload, list, delete) and configuration controls (model choice, retrieval params).

## Tech Stack
- **Language:** Python 3.10+
- **Frameworks:** Flask or Streamlit for the web UI
- **Data & Retrieval:** FAISS / Chroma / Supabase pgvector
- **LLM APIs:** OpenAI GPT-4 Turbo & GPT-3.5 Turbo, `text-embedding-3` models
- **Parsing Libraries:** `pdfplumber`, `python-docx`, `markdown`, `pypdf`
- **Utilities:** `python-dotenv`, `pydantic`, `typer` for CLI tooling, `pytest`

## Learning Roadmap
### 1. Python Foundations
- Control flow (`if`, `for`, `while`), functions, list/dict comprehensions
- Working with packages: virtual environments, `pip`, project structure
- File I/O for reading documents, handling encodings, basic error handling
- Standard library modules useful for this project (`pathlib`, `json`, `logging`, `argparse`/`typer`)

### 2. Environment & Tooling
- Git basics (branching, committing) and GitHub workflow
- Using `.env` files and environment variables securely
- Dependency management with `requirements.txt` or `pip-tools`

### 3. Semantic Search & Embeddings
- Concept of vector embeddings and cosine similarity
- OpenAI embedding endpoints and rate limits
- Vector stores (FAISS, Chroma, pgvector) fundamentals
- Chunking strategies: window size, overlap, metadata tracking

### 4. Large Language Models & Prompting
- Difference between system, user, assistant messages
- Techniques for grounding answers and avoiding hallucinations
- Cost management: token budgeting, model choice trade-offs
- Basic prompt engineering patterns (instruction following, citations)

### 5. Document Ingestion
- Parsing PDFs with `pdfplumber`/`pypdf` and handling edge cases
- Reading DOCX, Markdown, and plain text files
- Text cleaning: removing headers/footers, normalizing whitespace, token estimation

### 6. Web Framework Basics
- Streamlit layout primitives or Flask routing/templates (choose your preferred path)
- Handling form submissions/uploads safely
- Rendering chat interfaces and streaming responses

### 7. Deployment & Operations
- Packaging the app with Docker (optional) or preparing for Render/Railway
- Managing secrets in production environments
- Monitoring logs, handling retries, and setting sensible timeouts

### 8. Responsible & Reliable AI
- Reading OpenAI usage policies and moderation guidelines
- Implementing guardrails, refusal behavior, and user feedback loops
- Evaluating QA quality with manual or automated checks

## Repository Structure (proposed)
```
.
├── README.md
├── app/                # Flask or Streamlit front end
│   ├── __init__.py
│   ├── main.py         # Entrypoint for UI
│   └── components/     # UI submodules
├── ingestion/          # Document loaders & chunkers
│   ├── loaders.py
│   └── chunker.py
├── retrieval/          # Vector store and search logic
│   ├── embeddings.py
│   └── store.py
├── llm/                # Prompting and answer orchestration
│   └── qa_chain.py
├── data/
│   ├── raw/            # Uploaded source docs (gitignored)
│   └── processed/      # Chunked text + embeddings (gitignored)
├── configs/
│   └── settings.example.env
├── scripts/            # CLI helpers (ingest, evaluate, etc.)
└── tests/
    └── test_end_to_end.py
```

## Getting Started
1. **Clone & Initialize**
   ```bash
   git clone <repo-url>
   cd Q-A-Chatbot-Using-OpenAI-API
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Configure Environment**
   - Copy `configs/settings.example.env` to `.env`.
   - Set required variables:
     - `OPENAI_API_KEY`
     - `EMBEDDING_MODEL` (`text-embedding-3-small` by default)
     - `CHAT_MODEL` (`gpt-4-turbo` or `gpt-3.5-turbo`)
     - Optional: `VECTOR_STORE_URL`, `VECTOR_STORE_API_KEY`, `APP_ENV`
3. **Prepare Document Storage**
   - Place source files in `data/raw/` (this folder should be excluded from version control).
   - Run the ingestion script to parse and index documents (see below).
4. **Run the App**
   - For Streamlit: `streamlit run app/main.py`
   - For Flask: `flask --app app.main run --debug`

## Ingestion Workflow
```bash
python scripts/ingest.py \
  --input data/raw \
  --output data/processed \
  --chunk-size 500 \
  --overlap 50 \
  --vector-store faiss
```
- Parses supported document types, normalizes text, and creates overlapping chunks.
- Generates embeddings and upserts them into the configured vector store.
- Persists metadata JSON for auditing citations.

## Retrieval & Answering
1. User submits a question.
2. System embeds the question and retrieves top matches (e.g., top 5 chunks).
3. GPT receives a structured prompt with:
   - System instructions (stay grounded, cite sources, admit uncertainty).
   - Retrieved context blocks.
   - Conversation history (limited for token efficiency).
4. Model returns an answer plus references; UI renders the reply and links to source snippets.

## Testing Strategy
- **Unit tests:** Validate loaders, chunker logic, embedding adapters, prompt builders.
- **Integration tests:** Mock OpenAI API to ensure ingestion and retrieval pipeline works end-to-end.
- **Evaluation harness:** Optional notebook or script to benchmark answer quality on a labeled Q&A set.

## Deployment Ideas
- **Local demo:** Streamlit app for quick sharing.
- **Web deployment:** Render, Railway, or Fly.io for Flask, remembering to set environment variables securely.
- **Containerization:** Dockerfile with multi-stage build (Python runtime + system deps for PDF parsing).

## Roadmap
- Add UI document uploader and progress feedback
- Implement per-user workspaces & authentication
- Add conversation memory persistence (SQLite or Supabase)
- Integrate moderation or guardrails for unsafe prompts
- Log analytics: query volume, response latency, feedback scores
- Support additional file formats (HTML, CSV, Notion exports)

## Contributing
1. Fork the repo and create a feature branch.
2. Run existing tests (`pytest`) before submitting changes.
3. Open a pull request describing the feature, testing steps, and screenshots if UI-related.

## License
Choose a license that matches your sharing goals (MIT, Apache-2.0, etc.). Update this section once a decision is made.

## Additional Resources
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)
- [FAISS Documentation](https://faiss.ai/index.html)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Flask Quickstart](https://flask.palletsprojects.com/en/latest/quickstart/)
