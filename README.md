# DocuMind : High Precision RAG for PDFs

This is a sophisticated RAG (Retrieval-Augmented Generation) application designed with a minimalist, editorial aesthetic inspired by Claude. It allows users to upload multiple PDF documents and engage in context-aware conversations, receiving precise answers with full source citations and relevance ranking.



## 🚀 Features

- **Minimalist Editorial UI**: A clean, "paper-mode" interface designed for focus and readability.
- **Multi-document Upload**: Process up to 5 PDF documents simultaneously.
- **Conversational Intelligence**: Powered by **Gemini 2.5 Flash** for high-speed, high-accuracy reasoning.
- **Source Citation & Relevance**: Answers include direct links to original documents/pages with cross-encoder re-ranking scores.
- **Local LLM Support**: Option to run locally using LlamaCpp for privacy-centric environments.
- **Robust Dockerization**: Optimized Python 3.11 environment with healthchecks and build-time resilience.

## 🛠️ Tech Stack

- **Frontend**: Streamlit (Customized with Claude-style CSS)
- **Document Processing**: `pdfplumber`
- **Vector Store**: FAISS
- **Embeddings**: HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`)
- **Language Models**: 
  - **Cloud**: Google Gemini 2.5 Flash (via `langchain-google-genai`)
  - **Local**: Mistral-7B-Instruct (via `LlamaCpp`)
- **Re-ranking**: `sentence-transformers/ms-marco-MiniLM-L-6-v2` (Cross-Encoder)

## 🧩 Architecture Overview

The application follows a modular RAG pipeline:

1. **Document Processing**: Extracts text and tables using `pdfplumber`, cleaning headers and footers for better retrieval quality.
2. **Vectorization**: Chunks text using `RecursiveCharacterTextSplitter` and generates embeddings stored in a local FAISS index.
3. **Hybrid Retrieval**: Combines standard vector search with a **Cross-Encoder re-ranking** step to ensure the most relevant context is passed to the LLM.
4. **Conversational Memory**: Maintains state using LangChain's `ConversationBufferMemory` to handle multi-turn questions effectively.

## 🚦 Getting Started

### 1. Clone the Repository
```bash
git clone DocuMind-High-Precision-RAG-for-PDFs
cd DocuMind-High-Precision-RAG-for-PDFs
```

### 2. Configure Environment Variables
Create a `.env` file in the project root:
```bash
GOOGLE_API_KEY=your_gemini_api_key_here

```

### 3. Run with Docker (Recommended)
The updated Docker environment uses Python 3.11 and includes all necessary system tools for FAISS and LlamaCpp.
```bash
docker compose up --build
```

### 4. Local Installation
```bash
python -m venv venv
source venv/bin/activate # or venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py
```

## 📌 Usage
1. **Library**: Upload PDFs in the sidebar to populate your knowledge base.
2. **Interact**: Ask questions in the centered chat interface.
3. **Verify**: Use the "View sources" expander to see exactly where the information came from and its relevance score.

## 🚧 Future Roadmap
- [ ] Support for DOCX and CSV formats.
- [ ] Integration with advanced metadata filtering.
- [ ] Exportable chat summaries and citation reports.

---
Built with ❤️ for focused research and discovery.
