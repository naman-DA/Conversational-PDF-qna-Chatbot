import streamlit as st
from langchain_chroma import Chroma 
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory 
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
load_dotenv()

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Setting up steramlit 
st.title("Conversational RAG with PDF uploads and Chat history")
st.write("Upload pdf and chat with their content")

api_key=st.text_input("Enter youd GROQ api key",type="password")

# check f api key is provided 
if api_key:
    llm=ChatGroq(groq_api_key=api_key,model_name="meta-llama/llama-4-scout-17b-16e-instruct")

    # chat interface
    session_id=st.text_input("session ID",value="default_session")

    #statefull chat history
    if 'store' not in st.session_state:
        st.session_state.store={}
    
    uploaded_files=st.file_uploader("choose a PDF file",type="pdf",accept_multiple_files=True)   

    if uploaded_files:
        document=[]
        
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
            
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name
                
            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            document.extend(docs)


        #spliit and create embeddings for the documents    
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=200)
        splits=text_splitter.split_documents(document)
        
        if "vectorstore" not in st.session_state:
            vectorstore = Chroma.from_documents(documents=splits,embedding=embeddings)
            st.session_state.vectorstore = vectorstore

        retriever = st.session_state.vectorstore.as_retriever()


system_prompt=(
    "you are an assistant for question-answer prompt"
    "use the following pieces of the retrieved context to answer"
    "the question . if you dont know the answer,say that you"
    "dont know. Use three sentences maximum and keep the "
    "answer concise"
    "\n\n"
    "{context}"
)

if api_key and uploaded_files:
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    def retrieve_docs(inputs):
        docs = retriever.invoke(inputs["input"])
        context = "\n\n".join(doc.page_content for doc in docs)
        
        return {
            "context": context,
            "input": inputs["input"],
            "chat_history": inputs["chat_history"],
        }
        
    retrieval_runnable = RunnableLambda(retrieve_docs)

    rag_pipeline = (
        RunnablePassthrough()
        | retrieval_runnable
        | qa_prompt
        | llm
)

    def get_session_history(session:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id]=ChatMessageHistory()
        return st.session_state.store[session_id]   
    
    conversational_rag_chain=RunnableWithMessageHistory(
        rag_pipeline,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
)

    user_input=st.text_input("Your Question")
    
    if user_input:
    # session_history=get_session_history=get_session_history(session_id)
        response=conversational_rag_chain.invoke(
            {"input":user_input},config={"configurable":{"session_id":session_id}}
        )
        
        st.write(st.session_state.store)
        st.success(response.content)
        st.write("Chat history")
        
        for msg in get_session_history(session_id).messages:
            st.write(f"{msg.type}:{msg.content}")
else:
    st.warning("Please enter your key")