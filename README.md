# 📄 Conversational PDF Q&A Chatbot

A conversational Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and ask natural-language questions about their content. The application retrieves relevant information from uploaded documents and generates concise, context-grounded responses while maintaining conversation history.

## 🚀 Live Demo

[Try the Live Demo](https://conversational-pdf-qna-chatbot-uqfsbcslpbbdeesksenuwg.streamlit.app/)

## 💻 GitHub Repository

[View Source Code](https://github.com/naman-DA/Conversational-PDF-qna-Chatbot)

---

## ✨ Features

- Upload one or multiple PDF documents
- Extract text from PDF documents using PyPDF
- Recursive document chunking
- HuggingFace embeddings using `all-MiniLM-L6-v2`
- Semantic similarity search using ChromaDB
- Retrieval-Augmented Generation (RAG)
- Context-grounded question answering
- Conversational chat history
- Multi-turn questions and follow-up queries
- Streamlit chat interface
- Secure Groq API key management using Streamlit Secrets
- Local `.env` support for development
- Automatic vector-store rebuilding when uploaded documents change

---

## 🏗️ Architecture

PDF Upload
    ↓
PyPDFLoader
    ↓
RecursiveCharacterTextSplitter
    ↓
HuggingFace Embeddings
    ↓
ChromaDB Vector Store
    ↓
Semantic Retriever
    ↓
Retrieved Context
    ↓
LangChain RAG Pipeline
    ↓
GPT-OSS-120B via Groq
    ↓
Grounded Response

---

## 🔄 RAG Workflow

### 1. PDF Ingestion

Users upload one or multiple PDF documents through the Streamlit interface. `PyPDFLoader` extracts the document text and metadata.

### 2. Text Chunking

The extracted documents are split into smaller chunks using `RecursiveCharacterTextSplitter`.

- Chunk Size: 5000
- Chunk Overlap: 200

### 3. Embedding Generation

Each document chunk is converted into a vector representation using the HuggingFace `all-MiniLM-L6-v2` embedding model.

### 4. Vector Storage

The generated embeddings are stored in ChromaDB, enabling semantic similarity-based document retrieval.

### 5. Semantic Retrieval

When the user asks a question, the retriever searches the vector store and returns the most relevant document chunks.

### 6. Context Injection

The retrieved document chunks are passed to the prompt as context. The system prompt instructs the model to answer using the retrieved information and avoid inventing information when the answer is unavailable.

### 7. Response Generation

The retrieved context, user question, and conversation history are passed through the LangChain RAG pipeline to GPT-OSS-120B through Groq.

### 8. Conversational History

`RunnableWithMessageHistory` maintains conversation history, allowing the chatbot to handle follow-up questions within the same session.

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Programming Language | Python |
| UI | Streamlit |
| LLM Framework | LangChain |
| LLM | GPT-OSS-120B |
| LLM Provider | Groq |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Database | ChromaDB |
| PDF Processing | PyPDF |
| Environment Management | python-dotenv |
| Version Control | Git, GitHub |
| Deployment | Streamlit Community Cloud |

---

## 📁 Project Structure

Conversational-PDF-qna-Chatbot/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .vscode/

---

## ⚙️ Local Setup

### 1. Clone the Repository

    git clone https://github.com/naman-DA/Conversational-PDF-qna-Chatbot.git

### 2. Navigate to the Project

    cd Conversational-PDF-qna-Chatbot

### 3. Create a Virtual Environment

Windows:

    python -m venv venv

Activate the environment:

    venv\Scripts\activate

### 4. Install Dependencies

    pip install -r requirements.txt

### 5. Configure the Groq API Key

Create a `.env` file in the project root:

    GROQ_API_KEY=your_groq_api_key

Never commit your `.env` file to GitHub.

### 6. Run the Application

    streamlit run app.py

The application will be available at:

    http://localhost:8501

---

## ☁️ Deployment

The application is deployed using Streamlit Community Cloud.

The Groq API key is configured through Streamlit Secrets:

    GROQ_API_KEY = "your_groq_api_key"

The API key is not stored in the source code or GitHub repository.

---

## 💬 Example Usage

1. Open the live application.
2. Upload one or more PDF documents.
3. Ask a question about the uploaded content.
4. The application retrieves relevant document chunks.
5. The retrieved context is passed to the LLM.
6. The chatbot generates a concise, context-grounded answer.
7. Ask follow-up questions to continue the conversation.

Example:

User:
What is the main purpose of this system?

Assistant:
The chatbot retrieves relevant information from the uploaded PDF and generates an answer based on the retrieved context.

Follow-up:

User:
How does it work?

The chatbot uses the previous conversation history along with the newly retrieved context to answer the follow-up question.

---

## 🧠 Key Concepts Demonstrated

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Vector Embeddings
- Vector Databases
- Document Chunking
- Context Retrieval
- Prompt Engineering
- Conversational Memory
- LLM Integration
- LangChain Runnable Pipelines
- PDF Document Processing
- Streamlit Application Development
- LLM API Integration
- Cloud Deployment

---

## 🔍 Challenges & Solutions

### Irrelevant Retrieved Context

During development, irrelevant document chunks could sometimes be retrieved. Retrieval quality was improved by tuning the chunk size and chunk overlap so that the retriever could provide more meaningful sections of the document to the LLM.

### LLM Hallucination

The system prompt instructs the model to rely on the retrieved document context and respond that it does not know when the required information is not available in the retrieved context.

### API Key Security

The Groq API key is not hardcoded into the application.

For local development, the key is loaded from `.env`.

For the deployed application, the key is stored using Streamlit Secrets.

### Multiple PDF Handling

When the uploaded PDF set changes, the application rebuilds the ChromaDB vector store using the newly uploaded documents and clears the previous conversation state.

---

## 🚀 Future Improvements

- Add source and page citations to generated answers
- Add configurable retrieval parameters
- Add document-specific chat sessions
- Add persistent vector-store management
- Add streaming LLM responses
- Add document preview and metadata display
- Improve retrieval using hybrid search
- Add retrieval and response evaluation metrics
- Add automated RAG evaluation

---

## 👨‍💻 Author

Naman Garg

B.Tech Computer Science

GitHub: https://github.com/naman-DA

LinkedIn: https://www.linkedin.com/in/naman-garg-16672b327

---

## 🔗 Project Links

Live Demo:
https://conversational-pdf-qna-chatbot-uqfsbcslpbbdeesksenuwg.streamlit.app/

GitHub Repository:
https://github.com/naman-DA/Conversational-PDF-qna-Chatbot
