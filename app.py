import os
import re
import streamlit as st
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS

from sentence_transformers import CrossEncoder

from components.pdf_handler import process_multiple_pdfs
from components.llm_handler import get_llm
from components.memory_handler import get_memory

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 100


if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="./models/embeddings"
    )


@st.cache_resource
def get_cross_encoder():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )


def normalize_section_headers(text: str) -> str:
    text = re.sub(r'\n(?=\d+\.\s+)', '\n\n', text)
    text = re.sub(r'\n(?=(Introduction|Abstract|Conclusion|References|Appendix))',
                  r'\n\n', text, flags=re.IGNORECASE)
    return text


def create_qa_chain(documents, embedding_model):
    text_splitter = get_text_splitter()

    for doc in documents:
        doc.page_content = normalize_section_headers(doc.page_content)

    splits = text_splitter.split_documents(documents)

    if not splits:
        return None

    vector_store = FAISS.from_documents(
        documents=splits,
        embedding=embedding_model
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=get_llm(use_cloud=True),
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        memory=get_memory(),
        return_source_documents=True,
        output_key="answer",
    )

    return qa_chain


def format_chat_history(chat_history: list) -> str:
    return "\n".join([f"User: {q}\nAssistant: {r}" for q, r, a, _ in chat_history])


def re_rank_results(query, cross_encoder):
    docs = st.session_state.qa_chain.retriever.invoke(query)

    pairs = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)
    ranked_docs_with_scores = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )
    top_docs_with_scores = ranked_docs_with_scores[:3]

    context = "\n\n".join(
        [doc.page_content for _, doc in top_docs_with_scores])
    prompt = f"""You are a helpful assistant that answers questions based on the provided document excerpts and chat history.\n\n
    Chat History:\n{format_chat_history(st.session_state.chat_history)}\n\n
    Context:\n{context}\n\nQuestion:\n{query} \n\n
    Answer in a clear and concise manner using the most relevant context and chat history. Use the chat history for extracting entities and aspects.
    """
    print(prompt)
    result = st.session_state.qa_chain.invoke({"question": prompt})

    return result, top_docs_with_scores


def main():
    st.set_page_config(
        page_title="AI-Powered Document Q&A",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Injecting Custom Claude-style CSS
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,500;8..60,600&display=swap');

        /* Global Typography */
        html, body, [class*="st-"] {
            font-family: 'Inter', sans-serif;
            color: #2C2C2A;
        }
        
        h1, h2, h3, .stTitle {
            font-family: 'Source Serif 4', serif !important;
            font-weight: 500 !important;
            color: #1A1A18 !important;
        }

        /* App Background */
        .stApp {
            background-color: #F9F9F8;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #F0EFE9 !important;
            border-right: 1px solid rgba(0,0,0,0.05);
        }
        
        [data-testid="stSidebar"] .stMarkdown h2 {
            font-size: 1.2rem;
            margin-top: 2rem;
            color: #6B6B66;
        }

        /* Chat Input Styling */
        .stChatInput {
            bottom: 2rem !important;
            max-width: 800px !important;
            margin: 0 auto !important;
            border-radius: 12px !important;
            border: 1px solid rgba(0,0,0,0.1) !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
            background: white !important;
        }

        /* Chat Message Styling */
        [data-testid="stChatMessage"] {
            background-color: transparent !important;
            border: none !important;
            padding: 1.5rem 0 !important;
            max-width: 800px !important;
            margin: 0 auto !important;
        }
        
        /* Assistant Message background like Claude */
        [data-testid="stChatMessageAssistant"] {
            background-color: rgba(217, 119, 87, 0.03) !important;
            border-radius: 16px !important;
            padding: 1.5rem !important;
            margin-bottom: 1rem !important;
        }

        .stChatMessageContent {
            font-family: 'Inter', sans-serif;
            line-height: 1.6;
            font-size: 1.05rem;
        }

        /* Hide Streamlit elements for cleaner look */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Buttons */
        .stButton button {
            border-radius: 8px !important;
            border: 1px solid rgba(0,0,0,0.1) !important;
            background-color: white !important;
            color: #2C2C2A !important;
            font-weight: 500 !important;
            transition: all 0.2s ease;
        }
        
        .stButton button:hover {
            border-color: #D97757 !important;
            color: #D97757 !important;
            background-color: rgba(217, 119, 87, 0.02) !important;
        }

        /* Expander */
        .stExpander {
            border: none !important;
            background-color: white !important;
            border-radius: 12px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.03) !important;
            margin-top: 1rem;
        }
        
        /* Centered Landing Page */
        .landing-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 60vh;
            text-align: center;
            max-width: 700px;
            margin: 0 auto;
        }
        
        .landing-title {
            font-family: 'Source Serif 4', serif;
            font-size: 3rem;
            margin-bottom: 1rem;
            color: #1A1A18;
        }
        
        .landing-subtitle {
            font-size: 1.2rem;
            color: #6B6B66;
            margin-bottom: 2rem;
        }

        </style>
    """, unsafe_allow_html=True)

    # Sidebar Header
    with st.sidebar:
        st.markdown("<h2 style='margin-top:0;'>Library</h2>", unsafe_allow_html=True)
        
        if st.button("＋ New Chat", use_container_width=True):
            st.session_state.chat_history = []
            if st.session_state.get("qa_chain"):
                st.session_state.qa_chain.memory.clear()
            st.rerun()
        
        st.divider()

    embedding_model = get_embedding_model()
    cross_encoder = get_cross_encoder()

    with st.sidebar:
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDFs to start querying", type="pdf", accept_multiple_files=True, label_visibility="collapsed")

        if len(uploaded_files) > 5:
            st.warning("Please upload a maximum of 5 files at a time.")
            uploaded_files = uploaded_files[:5]

        if uploaded_files and "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = uploaded_files
            st.session_state.processed_documents = None

    if uploaded_files:
        try:
            if st.session_state.processed_documents is None:
                with st.spinner('Processing documents...'):
                    documents = process_multiple_pdfs(uploaded_files)

                    if not documents:
                        st.warning("No text found in the uploaded files.")
                        return

                    st.session_state.processed_documents = documents

            documents = st.session_state.processed_documents

            if "qa_chain" not in st.session_state or st.session_state.qa_chain is None:
                st.session_state.qa_chain = create_qa_chain(
                    documents, embedding_model)

            if st.session_state.qa_chain:
                question = st.chat_input(
                    "Ask a question about your document...")
                if question:
                    with st.spinner("Thinking..."):
                        # result = st.session_state.qa_chain.invoke({"question": question})
                        result, top_docs_with_scores = re_rank_results(
                            question, cross_encoder)
                        raw_answer = result["answer"]
                        answer_text = raw_answer
                        sources = result.get("source_documents", [])
                        if sources:
                            cited_docs = ", ".join(set(doc.metadata.get(
                                "doc_name", "Unknown") for doc in sources))
                            answer_text += f"\n\n**Sources cited**: {cited_docs}"

                        st.session_state.chat_history.append(
                            (question, raw_answer, answer_text, top_docs_with_scores))

                for q, r, answer_text, msg_sources_with_scores in st.session_state.chat_history:
                    st.chat_message("user").markdown(q)
                    st.chat_message("assistant").markdown(answer_text)

                    if msg_sources_with_scores:
                        with st.expander("View sources and relevance scores"):
                            for i, (score, doc) in enumerate(msg_sources_with_scores):
                                doc_name = doc.metadata.get(
                                    "doc_name", "Unknown")
                                page = doc.metadata.get("page", "N/A")
                                st.markdown(
                                    f"**Source {i+1}:** {doc_name} (Page {page}) — Score: `{score:.2f}`")
                                st.code(
                                    doc.page_content[:200].strip() + "...", language="markdown")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.markdown("""
            <div class="landing-container">
                <div class="landing-title">How can I help you today?</div>
                <div class="landing-subtitle">Upload your documents in the sidebar to start a conversation with your PDF library.</div>
            </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
