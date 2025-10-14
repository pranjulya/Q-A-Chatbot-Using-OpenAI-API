# Q&A Chatbot Using OpenAI API

An open-source, retrieval-augmented Q&A chatbot that grounds answers in your own documents. The project demonstrates how to combine document parsing, embeddings, vector search, and GPT models inside a lightweight Streamlit interface and Typer-powered CLI.

## Key Features
- Document ingestion pipeline with PDF, DOCX, Markdown, and plain text support
- Deterministic text chunking with adjustable overlap to preserve context
- Local JSON-based vector store for simple demos and teaching environments
- OpenAI embeddings + chat completions for answer generation with citations
- Streamlit web UI for conversational querying
- Fully open-source with MIT license, contribution guide, and code-of-conduct

## Quick Start
1. **Clone and install dependencies**
   ```bash
   git clone <repo-url>
   cd Q-A-Chatbot-Using-OpenAI-API
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Configure environment variables**
   ```bash
   cp configs/settings.example.env .env
   # edit .env with your OpenAI API key and (optionally) custom model names
   ```
3. **Load documents and build the vector store**
   ```bash
   mkdir -p data/raw
   # add .txt/.md/.pdf/.docx files to data/raw
   python scripts/ingest.py data/raw --output data/processed/index.json
   ```
4. **Launch the Streamlit app**
   ```bash
   streamlit run app/main.py
   ```

## Usage
- Point the chatbot at a folder of course notes, handbooks, or knowledge base articles.
- Re-run `scripts/ingest.py` whenever documents change.
- In the Streamlit sidebar, configure embedding/chat models and the number of context chunks.
- Answers cite their source files. When context is insufficient, the model is instructed to acknowledge uncertainty.
- If no relevant chunks are retrieved from the vector store, the assistant explicitly replies that it cannot answer yet.

## Project Structure
```
.
├── app/                   # Streamlit UI
│   ├── __init__.py
│   └── main.py
├── configs/               # Environment templates
│   └── settings.example.env
├── ingestion/             # Loaders and chunking utilities
│   ├── chunker.py
│   └── loaders.py
├── llm/                   # Prompt orchestration
│   └── qa_chain.py
├── retrieval/             # Embeddings + lightweight vector store
│   ├── embeddings.py
│   └── store.py
├── scripts/               # CLI entry points
│   └── ingest.py
├── tests/                 # Pytest-based smoke tests
│   ├── test_chunker.py
│   └── test_store.py
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── LICENSE (MIT)
├── README.md
└── requirements.txt
```

## Learning Roadmap
The repository doubles as a companion curriculum. Suggested topics for students:

1. **Python Foundations** – Control flow, functions, comprehensions, virtual environments, file I/O, logging.
2. **Environment & Tooling** – Git/GitHub workflow, `.env` management, dependency handling.
3. **Semantic Search & Embeddings** – Concept of vector similarity, OpenAI embedding APIs, vector store design.
4. **Prompting & LLMs** – System/user messages, grounding answers, limiting hallucinations, prompt engineering.
5. **Document Ingestion** – Parsing PDFs/DOCX/Markdown, cleaning text, chunk sizing, metadata for citations.
6. **Web Apps** – Streamlit layout/components, handling uploads, rendering chat-style responses.
7. **Deployment & Ops** – Packaging with Docker (optional), hosting on Render/Railway, managing secrets, logging.
8. **Responsible AI** – Usage policies, moderation, refusal behavior, evaluation of answer quality.

## Testing
Run the fast unit tests with:
```bash
pytest
```
Tests cover core utilities (chunking and vector-store retrieval). Expand them as the project grows.

## Contributing
Contributions from students and educators are welcome! Please read `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md` before opening an issue or pull request.

## License
Released under the MIT License. See `LICENSE` for details.

## Acknowledgements
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Typer Documentation](https://typer.tiangolo.com/)
