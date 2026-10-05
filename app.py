import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableWithMessageHistory,
)
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Streamlit UI

st.set_page_config(
    page_title="Conversational PDF Q&A",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Conversational RAG with PDF Chat")
st.write("Upload PDF documents and ask questions about their content.")

if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
else:
    api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error(
        "Groq API key is not configured. "
        "Add GROQ_API_KEY to Streamlit Secrets or your local .env file."
    )
    st.stop()

llm = ChatGroq(
    groq_api_key=api_key,
    model_name="openai/gpt-oss-120b",
    temperature=0,
)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
# Session State

if "store" not in st.session_state:
    st.session_state.store = {}

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []

# Session ID

session_id = st.text_input(
    "Session ID",
    value="default_session",
)

# PDF Upload

uploaded_files = st.file_uploader(
    "Choose PDF file(s)",
    type="pdf",
    accept_multiple_files=True,
)

# Process PDFs

if uploaded_files:
    current_file_names = sorted(
        [uploaded_file.name for uploaded_file in uploaded_files]
    )

    # Rebuild vector store when uploaded PDFs change
    if current_file_names != st.session_state.uploaded_file_names:

        documents = []

        for uploaded_file in uploaded_files:

            temp_pdf = f"./{uploaded_file.name}"

            with open(temp_pdf, "wb") as file:
                file.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()

            documents.extend(docs)

            # Remove temporary PDF after loading
            try:
                os.remove(temp_pdf)
            except OSError:
                pass

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=200,
        )

        splits = text_splitter.split_documents(documents)

        # Create ChromaDB vector store

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
        )

        st.session_state.vectorstore = vectorstore
        st.session_state.uploaded_file_names = current_file_names

        # Clear previous chat when documents change
        st.session_state.store = {}

        st.success(
            f"Processed {len(uploaded_files)} PDF file(s) successfully."
        )
# RAG Pipeline

if st.session_state.vectorstore is not None:

    retriever = st.session_state.vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Keep the answer concise and grounded in the "
        "provided context. Do not invent information."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Retrieval Function

    def retrieve_docs(inputs):

        docs = retriever.invoke(inputs["input"])

        context = "\n\n".join(
            doc.page_content for doc in docs
        )

        return {
            "context": context,
            "input": inputs["input"],
            "chat_history": inputs["chat_history"],
        }


    retrieval_runnable = RunnableLambda(retrieve_docs)

    # RAG Pipeline

    rag_pipeline = (
        RunnablePassthrough()
        | retrieval_runnable
        | qa_prompt
        | llm
    )

    # Chat History

    def get_session_history(
        session: str,
    ) -> BaseChatMessageHistory:

        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()

        return st.session_state.store[session]


    conversational_rag_chain = RunnableWithMessageHistory(
        rag_pipeline,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )


    # Chat Input

    user_input = st.chat_input(
        "Ask a question about your PDF..."
    )

    # Process User Question

    if user_input:

        conversational_rag_chain.invoke(
            {"input": user_input},
            config={
                "configurable": {
                    "session_id": session_id
                }
            },
        )

    # Display Chat History

    if session_id in st.session_state.store:

        for message in get_session_history(session_id).messages:

            if message.type == "human":

                with st.chat_message("user"):
                    st.markdown(message.content)

            elif message.type == "ai":

                with st.chat_message("assistant"):
                    st.markdown(message.content)
else:
    st.info(
        "Upload at least one PDF to start chatting."
    )