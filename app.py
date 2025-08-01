import streamlit as st
import os
from dotenv import load_dotenv
from typing import Optional

from rag_pipeline import RAGPipeline, validate_google_api_key


def initialize_session_state() -> None:
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False
    if 'api_key_validated' not in st.session_state:
        st.session_state.api_key_validated = False


def load_api_key() -> Optional[str]:
    load_dotenv()
    return os.getenv('GOOGLE_API_KEY')


def setup_sidebar() -> Optional[str]:
    st.sidebar.title("🔧 Configuration")
    st.sidebar.subheader("Google API Key")

    env_api_key = load_api_key()

    if env_api_key:
        st.sidebar.success("✅ API key loaded from environment")
        api_key = env_api_key

        override_key = st.sidebar.text_input(
            "Override with your own key (optional):",
            type="password",
            help="Leave empty to use the environment key"
        )
        if override_key:
            api_key = override_key
    else:
        st.sidebar.warning("⚠️ No API key found in environment")
        api_key = st.sidebar.text_input(
            "Enter your Google API Key:",
            type="password",
            help="Get your API key from: https://makersuite.google.com/app/apikey"
        )

    if api_key and not st.session_state.api_key_validated:
        with st.spinner("Validating API key..."):
            is_valid, message = validate_google_api_key(api_key)
            if is_valid:
                st.session_state.api_key_validated = True
                st.sidebar.success("✅ API key validated successfully")
            else:
                st.sidebar.error(f"❌ {message}")
                return None
    elif api_key and st.session_state.api_key_validated:
        st.sidebar.success("✅ API key validated")

    if not api_key:
        st.sidebar.info("📝 Please enter your Google API key to continue")
        return None

    return api_key


def setup_file_uploader(api_key: str) -> None:
    st.sidebar.subheader("📄 Document Upload")

    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF documents",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF files to chat with"
    )

    if uploaded_files:
        st.sidebar.info(f"📁 {len(uploaded_files)} file(s) selected")
        if st.sidebar.button("🚀 Process Documents", type="primary"):
            process_documents(uploaded_files, api_key)

    if st.session_state.documents_processed:
        st.sidebar.success("✅ Documents processed successfully")

        if st.sidebar.button("🔄 Clear & Upload New Documents"):
            st.session_state.rag_pipeline = None
            st.session_state.chat_history = []
            st.session_state.documents_processed = False
            st.rerun()


def process_documents(uploaded_files, api_key: str) -> None:
    with st.spinner("🔄 Processing documents... This may take a few moments."):
        try:
            if st.session_state.rag_pipeline is None:
                st.session_state.rag_pipeline = RAGPipeline(api_key)

            success, message = st.session_state.rag_pipeline.process_documents(uploaded_files)

            if success:
                st.session_state.documents_processed = True
                st.session_state.chat_history = []
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")

        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")


def display_chat_interface() -> None:
    st.title("🤖 RAG Chatbot with Google Gemini")
    st.markdown("---")

    if not st.session_state.documents_processed:
        st.info("👈 Please upload and process PDF documents using the sidebar to start chatting!")
        return

    # Display chat history first
    display_chat_history()

    # Chat input should be at the top level
    question = st.chat_input("Ask a question about your documents...")

    if question:
        handle_user_question(question)


def handle_user_question(question: str) -> None:
    if not st.session_state.rag_pipeline or not st.session_state.rag_pipeline.is_ready():
        st.error("❌ Please process documents first!")
        return

    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.spinner("🤔 Thinking..."):
        success, answer, source_docs = st.session_state.rag_pipeline.get_response(question)

    if success:
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.session_state.chat_history = st.session_state.rag_pipeline.get_chat_history()
    else:
        st.error(f"❌ {answer}")
        st.session_state.chat_history.pop()


def display_chat_history() -> None:
    if not st.session_state.chat_history:
        st.info("💬 Start a conversation by asking a question about your documents!")
        return

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def display_usage_tips() -> None:
    with st.expander("💡 Usage Tips", expanded=False):
        st.markdown("""
        **How to use this RAG Chatbot:**
        
        1. **API Key**: Enter your Google API key in the sidebar  
        2. **Upload Documents**: Upload one or more PDFs  
        3. **Process**: Click "Process Documents"  
        4. **Chat**: Ask questions about your PDFs  

        **Features:**
        - 🧠 Conversational memory  
        - 📚 Multi-document support  
        - 🔍 Semantic search  
        - ⚡ Google Gemini LLM  

        **Tips:**
        - Ask specific questions  
        - Reference previous chat context  
        - Use well-formatted PDFs  
        """)


def main() -> None:
    st.set_page_config(
        page_title="RAG Chatbot with Gemini",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    initialize_session_state()

    api_key = setup_sidebar()

    if api_key:
        setup_file_uploader(api_key)

    # Chat interface must be at top level (for st.chat_input to work)
    display_chat_interface()

    # Side column content (not nested with chat_input)
    col1, col2 = st.columns([3, 1])

    with col2:
        display_usage_tips()
        st.markdown("---")
        st.markdown("**Powered by:**")
        st.markdown("- 🤖 Google Gemini 1.5 Pro")
        st.markdown("- 🔗 LangChain")
        st.markdown("- 📊 ChromaDB")
        st.markdown("- ⚡ Streamlit")

        if st.session_state.documents_processed and st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History", help="Clear the conversation history"):
                if st.session_state.rag_pipeline:
                    st.session_state.rag_pipeline.clear_chat_history()
                st.session_state.chat_history = []
                st.rerun()


if __name__ == "__main__":
    main()
