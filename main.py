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

# --- Minimal CSS for aesthetics (optional) ---
CUSTOM_CSS = """
<style>
.stApp {
    background-color: #121212;
    color: #E0E0E0;
    font-family: 'Inter', sans-serif;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
    border-radius: 8px;
}
.stFileUploader {
    border: 2px dashed #4CAF50;
    border-radius: 12px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- Document Loading Functions ---
def load_documents(uploaded_file):
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
        return None

    docs = loader.load()
    os.remove(temp_filename)
    return docs

def process_documents(docs):
    if not docs:
        return None

    st.info(f"Loaded {len(docs)} page(s).")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_documents(docs)
    st.info(f"Split into {len(chunks)} text chunks.")

    try:
        vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
        st.success("Document ready! You can ask questions now.")
        return vector_store.as_retriever()
    except Exception as e:
        st.error(f"Vector store creation failed: {e}")
        return None

# --- Main App ---
def main():
    st.set_page_config(page_title="AI Document RAG", layout="wide")
    st.title("📄 AI-Powered Document Q&A")
    st.markdown("---")

    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- Sidebar ---
    with st.sidebar:
        st.header("Upload Document")
        uploaded_file = st.file_uploader("Select a file (PDF, DOCX, TXT)", type=["pdf","docx","txt"])
        if uploaded_file and st.button("Process Document"):
            docs = load_documents(uploaded_file)
            if docs:
                st.session_state.retriever = process_documents(docs)
                st.session_state.messages = []

        st.markdown("---")
        st.caption(f"LLM: {llm.model}")
        st.caption(f"Embeddings: {embeddings.model}")

    # --- Chat Interface ---
    if st.session_state.retriever:
        st.subheader("Ask a Question about the Document")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask me anything..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                placeholder.markdown("Searching and synthesizing answer...")
                try:
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=st.session_state.retriever,
                        return_source_documents=False
                    )
                    response = qa_chain.invoke({"query": prompt})
                    result = response.get("result", "Sorry, could not find an answer.")
                    placeholder.markdown(result)
                    st.session_state.messages.append({"role": "assistant", "content": result})
                except Exception as e:
                    placeholder.error(f"Error: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": str(e)})
    else:
        st.info("Please upload a document from the sidebar.")

if __name__ == "__main__":
    main()
