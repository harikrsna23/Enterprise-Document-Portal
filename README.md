# 🏢 Enterprise Document Portal

An AI-powered document intelligence platform that enables users to analyze PDFs,
compare documents, and chat with their documents using a RAG pipeline.

---

## 🚀 Features

- 📄 **Document Analysis** — Upload a PDF and get structured AI-powered analysis
- 🔍 **Document Comparison** — Compare two PDFs page-by-page for differences
- 💬 **Doc Chat** — Conversational RAG to ask questions about your documents
- 🔐 **Session-based storage** — Each upload is isolated in its own session

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| LLM | Groq (llama-3.1-8b-instant) |
| Embeddings | Google text-embedding-004 |
| Vector Store | FAISS (local) |
| AI Framework | LangChain |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Testing | pytest |

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/harikrsna23/Enterprise-Document-Portal.git
cd Enterprise-Document-Portal
```

### 2. Create virtual environment
```bash
uv venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
uv pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp .env.example .env
```

Add your API keys to `.env`:
```
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key
LLM_PROVIDER=groq
ENV=local
```

### 5. Run the app
```bash
uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000` in your browser.

---

## 🔑 API Keys

| Key | Where to Get | Cost |
|---|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com/keys) | Free |
| `GOOGLE_API_KEY` | [aistudio.google.com](https://aistudio.google.com/apikey) | Free |

---

## 🏗️ Architecture
```
User uploads PDF
      ↓
FastAPI Backend
      ↓
LangChain Document Processor
      ↓
Google Embeddings → FAISS Vector Store
      ↓
Groq LLM (llama-3.1-8b-instant)
      ↓
Analysis / Comparison / Chat Response
```

---

## 📁 Project Structure
```
├── api/              # FastAPI routes and main app
├── src/
│   ├── document_ingestion/   # PDF loading and processing
│   ├── document_analyzer/    # AI analysis logic
│   ├── document_compare/     # Document comparison
│   └── document_chat/        # Conversational RAG
├── config/           # YAML configuration
├── utils/            # Model loader, config loader
├── logger/           # Structured JSON logging
├── exception/        # Custom exceptions
├── static/           # Frontend assets
├── templates/        # HTML templates
├── tests/            # pytest test suite
├── Dockerfile        # Container definition
└── .github/workflows # CI/CD pipeline
```

---

## 🐳 Docker
```bash
docker build -t enterprise-document-portal .
docker run -p 8000:8000 --env-file .env enterprise-document-portal
```

---

## 📝 Course Project

Built as part of a guided course on LangChain and RAG systems.
Independently set up, debugged, configured and deployed.