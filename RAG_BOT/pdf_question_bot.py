import os
import time
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Page setup
st.set_page_config(page_title="PDF & Book QA Assistant (Gemini)", page_icon="📚", layout="wide")
st.title("📚 PDF & Book QA System (Powered by Gemini)")

# Initialize session state variables
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""

# --- Sidebar: API Configuration & File Upload ---
with st.sidebar:
    st.header("1. API Configuration")
    api_key = st.text_input("Enter Gemini API Key", type="password", value=st.session_state.gemini_api_key)
    gemini_api_key = api_key or os.getenv("GEMINI_API_KEY")

    st.header("2. Document Upload")
    uploaded_files = st.file_uploader(
        "Upload Books or PDFs", type=["pdf"], accept_multiple_files=True
    )

    if st.button("Process Documents"):
        if not gemini_api_key:
            st.error("Please enter a valid Gemini API Key.")
        elif not uploaded_files:
            st.error("Please upload at least one PDF file.")
        else:
            with st.spinner("Reading & parsing PDF files..."):
                all_docs = []

                for uploaded_file in uploaded_files:
                    # Save temporary file to disk for PyPDFLoader
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    try:
                        loader = PyPDFLoader(tmp_path)
                        docs = loader.load()

                        # Preserve file metadata
                        for doc in docs:
                            doc.metadata["source_file"] = uploaded_file.name

                        all_docs.extend(docs)
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000,
                    chunk_overlap=250
                )
                splits = text_splitter.split_documents(all_docs)
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="gemini-embedding-2-preview",
                    google_api_key=gemini_api_key
                )

                st.info(f"Indexing {len(splits)} chunks. Processing in small batches to prevent 429 rate limits...")


                batch_size = 25
                vectorstore = FAISS.from_documents(splits[:batch_size], embeddings)
                progress_bar = st.progress(min(batch_size, len(splits)) / len(splits))

                for i in range(batch_size, len(splits), batch_size):
                    time.sleep(2)
                    batch = splits[i:i + batch_size]
                    vectorstore.add_documents(batch)
                    progress_bar.progress(min(i + batch_size, len(splits)) / len(splits))
                st.session_state.vectorstore = vectorstore
                st.session_state.gemini_api_key = gemini_api_key
                st.success(f"Indexed {len(uploaded_files)} file(s) successfully!")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if user_query := st.chat_input("Ask a question about your uploaded documents..."):
    if not st.session_state.vectorstore:
        st.error("Please upload and process your documents in the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                retriever = st.session_state.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5}
                )

                prompt = ChatPromptTemplate.from_messages([
                    ("system",
                     "You are an intelligent document research assistant. Answer the user's "
                     "question thoroughly using ONLY the provided context snippets from the uploaded documents. "
                     "If the answer cannot be determined from the context, state clearly that the "
                     "information is not found in the documents.\n\n"
                     "Context:\n{context}"),
                    ("human", "{input}"),
                ])

                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=st.session_state.gemini_api_key,
                    temperature=0.2
                )
                combine_docs_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

                response = rag_chain.invoke({"input": user_query})
                answer = response["answer"]

                st.markdown(answer)

                # Context sources expander
                with st.expander("Retrieved Source Passages"):
                    for doc in response["context"]:
                        src = doc.metadata.get("source_file", "Unknown")
                        page = doc.metadata.get("page", 0) + 1
                        st.markdown(f"**Source:** `{src}` (Page {page})")
                        st.caption(doc.page_content[:250] + "...")

        st.session_state.messages.append({"role": "assistant", "content": answer})