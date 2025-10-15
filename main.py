import os
import io
import streamlit as st
from dotenv import load_dotenv

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY not found. Please set it in your .env file.")
    st.stop()

# Initialize Models
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")

# --- Custom CSS ---
CUSTOM_CSS = """
<style>
.stApp {background-color: #121212; color: #E0E0E0; font-family: 'Inter', sans-serif;}
#MainMenu, footer, header {visibility: hidden;}
.main .block-container {padding:2rem; max-width:1000px;}
[data-testid="stSidebar"] {background:#1E1E1E; border-right:1px solid #333;}
.stFileUploader {border:2px dashed #4CAF50; padding:1rem; border-radius:12px; background:#282828;}
.stButton>button {background:#4CAF50; color:white; font-weight:bold; border-radius:8px; padding:0.5rem 1rem; border:none; box-shadow:0 4px #388E3C; transition:all 0.2s;}
.stButton>button:hover {background:#5CB860; box-shadow:0 2px #388E3C; transform:translateY(2px);}
[data-testid="stChatMessage"] {border-radius:12px; padding:10px 15px; margin:10px 0; line-height:1.6;}
[data-testid="stChatMessage"]:has(.stChatMessageContent) > div:first-child {background:#2A2A2A; border-bottom-right-radius:2px !important;}
[data-testid="stChatMessage"]:has(.stChatMessageContent) > div:nth-child(2) {background:#242424; border-bottom-left-radius:2px !important;}
[data-testid="stForm"] > div:last-child > div:last-child {background:#1E1E1E; padding:10px; border-radius:10px;}
[data-testid="stTextInput"] > div > div > input {background:#282828; color:#E0E0E0; border:1px solid #3A3A3A; border-radius:8px; padding:10px;}
[data-testid="stAlert"] {border-radius:8px; box-shadow:0 2px 5px rgba(0,0,0,0.3);}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- Session State Initialization ---
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sidebar_state" not in st.session_state:
    st.session_state.sidebar_state = True  # True = open, False = closed

# --- Sidebar Toggle ---
def toggle_sidebar():
    st.session_state.sidebar_state = not st.session_state.sidebar_state

st.button("Toggle Sidebar", on_click=toggle_sidebar)

# Only show sidebar if open
if st.session_state.sidebar_state:
    with st.sidebar:
        st.header("Upload Document")
        st.caption("PDF, DOCX, TXT accepted.")
        uploaded_file = st.file_uploader(
            "Select your file",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=False,
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            if st.button("Process Document", use_container_width=True):
                with st.spinner("Loading and processing document..."):
                    # Load documents
                    file_extension = uploaded_file.name.split(".")[-1].lower()
                    temp_file = io.BytesIO(uploaded_file.read())
                    temp_filename = f"temp.{file_extension}"
                    with open(temp_filename, "wb") as f:
                        f.write(temp_file.getbuffer())

                    if file_extension == "pdf":
                        loader = PyPDFLoader(temp_filename)
                    elif file_extension == "docx":
                        loader = Docx2txtLoader(temp_filename)
                    elif file_extension == "txt":
                        loader = TextLoader(temp_filename)
                    else:
                        st.error(f"Unsupported file type: .{file_extension}")
                        os.remove(temp_filename)
                        loader = None

                    if loader:
                        docs = loader.load()
                        os.remove(temp_filename)
                        st.session_state.retriever = Chroma.from_documents(
                            documents=RecursiveCharacterTextSplitter(
                                chunk_size=1000, chunk_overlap=150
                            ).split_documents(docs),
                            embedding=embeddings,
                            persist_directory="./chroma_db"  # Optional persistence
                        ).as_retriever()
                        st.session_state.messages = []

        st.markdown("<hr style='border-top: 1px solid #3A3A3A;'>", unsafe_allow_html=True)
        st.caption(f"**LLM:** `{llm.model}`")
        st.caption(f"**Embeddings:** `{embeddings.model}`")
        st.caption("By Team 11.")

# --- Main Chat Interface ---
st.title("📄 AI-Powered Document Q&A (RAG)")
st.markdown("---")

if st.session_state.retriever:
    st.subheader("Ask a Question about the Document")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me anything about the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Searching the document and synthesizing an answer...")

            try:
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state.retriever,
                    return_source_documents=False
                )
                response = qa_chain.invoke({"query": prompt})
                full_response = response.get("result", "Sorry, I couldn't find an answer in the document.")
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                error_message = f"Error: {e}"
                message_placeholder.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
else:
    st.info("Please upload and process a document in the sidebar to begin chatting.")
