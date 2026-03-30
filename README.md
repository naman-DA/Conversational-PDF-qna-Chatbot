## Conversational RAG with PDF Uploads & Chat History

A Streamlit-based Conversational RAG (Retrieval-Augmented Generation) application that allows users to upload PDF files and interact with them through natural language queries using an LLM.

#### Features
- Upload multiple PDF documents
- Extract and chunk text using LangChain
- Generate embeddings with HuggingFace (all-MiniLM-L6-v2)
- Store vectors using ChromaDB
- Query documents using Groq LLM (LLaMA 4 Scout)
- Maintain session-based chat history
- Fast and interactive UI with Streamlit
- Tech Stack
  - Frontend/UI: Streamlit
  - LLM: Groq (LLaMA 4 Scout 17B)
  - Framework: LangChain
  - Embeddings: HuggingFace
  - Vector DB: Chroma
  - PDF Loader: PyPDFLoader

#### Project Architecture
User Query
   ↓
Retriever (Chroma Vector Store)
   ↓
Relevant PDF Context
   ↓
Prompt Template
   ↓
Groq LLM (LLaMA)
   ↓
Response + Chat History

#### How It Works
- User uploads one or more PDFs
- PDFs are loaded and split into chunks
- Embeddings are created and stored in ChromaDB
- User asks a question
- Relevant chunks are retrieved
- LLM generates a concise answer (max 3 sentences)
- Chat history is preserved using session-based memory

#### Key Components
- Text Splitter: RecursiveCharacterTextSplitter
- Retriever: Chroma Vector Store Retriever
- Prompt: Context-aware QA prompt
- Memory: RunnableWithMessageHistory
