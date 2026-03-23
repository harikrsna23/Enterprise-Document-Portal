import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise Document Portal",
    page_icon="🏢",
    layout="wide"
)

# ── Header ───────────────────────────────────────────────────
st.title("🏢 Enterprise Document Portal")
st.markdown("AI-powered document analysis, comparison and chat using RAG")
st.divider()

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_key = st.text_input("Groq API Key", 
                              value=os.getenv("GROQ_API_KEY", ""), 
                              type="password")
    google_key = st.text_input("Google API Key", 
                                value=os.getenv("GOOGLE_API_KEY", ""), 
                                type="password")
    st.divider()
    st.markdown("#### 🔑 Get Free API Keys")
    st.markdown("[Get Groq Key](https://console.groq.com/keys)")
    st.markdown("[Get Google Key](https://aistudio.google.com/apikey)")
    st.divider()
    st.markdown("#### 📦 Tech Stack")
    st.markdown("""
    - 🦜 LangChain
    - ⚡ Groq LLM
    - 🔍 FAISS Vector Store
    - 🧠 Google Embeddings
    - 🚀 Streamlit
    """)

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📄 Document Analysis", 
                              "🔍 Document Compare", 
                              "💬 Doc Chat"])

def get_model_loader():
    if not groq_key or not google_key:
        st.error("Please enter both API keys in the sidebar!")
        return None
    os.environ["GROQ_API_KEY"] = groq_key
    os.environ["GOOGLE_API_KEY"] = google_key
    os.environ["LLM_PROVIDER"] = "groq"
    os.environ["ENV"] = "local"
    try:
        from utils.model_loader import ModelLoader
        return ModelLoader()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# ── Tab 1: Document Analysis ─────────────────────────────────
with tab1:
    st.header("📄 Analyze a Document")
    st.markdown("Upload a PDF and get structured AI-powered analysis.")
    
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf", key="analyze")
    
    if uploaded_file and st.button("🔍 Run Analysis", type="primary"):
        with st.spinner("Analyzing document..."):
            try:
                loader = get_model_loader()
                if loader:
                    from src.document_ingestion.data_ingestion import DocHandler
                    from src.document_analyzer.data_analysis import DocumentAnalyzer
                    from utils.document_ops import FastAPIFileAdapter

                    with tempfile.TemporaryDirectory() as tmpdir:
                        file_adapter = FastAPIFileAdapter(uploaded_file)
                        handler = DocHandler(session_id="streamlit_analysis")
                        saved_path = handler.save_pdf(file_adapter)
                        text_content = handler.read_pdf(saved_path)

                        llm = loader.load_llm()
                        analyzer = DocumentAnalyzer()
                        if isinstance(text_content[0], str):
                                full_text = "\n".join(text_content)
                        else:
                            full_text = "\n".join([p.page_content for p in text_content])
                        result = analyzer.analyze_document(full_text)
                        st.success("Analysis Complete!")
                        st.markdown(result)
            except Exception as e:
                st.error(f"Analysis failed: {e}")

# ── Tab 2: Document Compare ───────────────────────────────────
with tab2:
    st.header("🔍 Compare Two Documents")
    st.markdown("Upload two PDFs to see page-wise differences.")
    
    col1, col2 = st.columns(2)
    with col1:
        ref_file = st.file_uploader("Reference PDF", type="pdf", key="ref")
    with col2:
        actual_file = st.file_uploader("Actual PDF", type="pdf", key="actual")
    
    if ref_file and actual_file and st.button("🔍 Compare", type="primary"):
        with st.spinner("Comparing documents..."):
            try:
                loader = get_model_loader()
                if loader:
                    from src.document_ingestion.data_ingestion import DocumentComparator
                    from src.document_compare.document_comparator import DocumentComparatorLLM
                    from utils.document_ops import FastAPIFileAdapter

                    ref_adapter = FastAPIFileAdapter(ref_file)
                    actual_adapter = FastAPIFileAdapter(actual_file)

                    comparator = DocumentComparator()
                    ref_path, actual_path = comparator.save_files(
                        ref_adapter, actual_adapter
                    )
                    docs = comparator.load_and_combine(ref_path, actual_path)

                    llm = loader.load_llm()
                    llm_comparator = DocumentComparatorLLM(llm=llm)
                    result = llm_comparator.compare(docs)

                    st.success("Comparison Complete!")
                    st.markdown(result)
            except Exception as e:
                st.error(f"Comparison failed: {e}")

# ── Tab 3: Doc Chat ───────────────────────────────────────────
with tab3:
    st.header("💬 Chat with your Document")
    st.markdown("Upload a PDF and ask questions about it.")

    chat_file = st.file_uploader("Upload PDF to chat with", 
                                  type="pdf", key="chat")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None

    if chat_file and st.button("📥 Load Document", type="primary"):
        with st.spinner("Loading document into vector store..."):
            try:
                loader = get_model_loader()
                if loader:
                    from src.document_ingestion.data_ingestion import ChatIngestor
                    from src.document_chat.retrieval import ConversationalRAG
                    from utils.document_ops import FastAPIFileAdapter

                    file_adapter = FastAPIFileAdapter(chat_file)
                    ingestor = ChatIngestor()
                    vectorstore = ingestor.ingest(
                        file_adapter,
                        embeddings=loader.load_embeddings()
                    )
                    llm = loader.load_llm()
                    rag = ConversationalRAG(llm=llm, vectorstore=vectorstore)
                    st.session_state.rag_chain = rag
                    st.session_state.chat_history = []
                    st.success("Document loaded! Ask me anything.")
            except Exception as e:
                st.error(f"Failed to load document: {e}")

    # Chat interface
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about your document..."):
        if not st.session_state.rag_chain:
            st.warning("Please upload and load a document first!")
        else:
            st.session_state.chat_history.append(
                {"role": "user", "content": prompt}
            )
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.rag_chain.chat(prompt)
                        st.markdown(response)
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": response}
                        )
                    except Exception as e:
                        st.error(f"Error: {e}")