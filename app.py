import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

# Set API keys
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# Initialize embeddings once
embeddings = OpenAIEmbeddings()

st.title("Conversational RAG with PDF Uploads and Chat History")
st.write("Upload PDFs and chat with their content")

# Input your Groq API key securely
api_key = st.text_input("Enter your Groq API key:", type="password")

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    # Session ID for multiple chat sessions
    session_id = st.text_input("Session ID", value="default_session")

    # Initialize session state containers
    if 'store' not in st.session_state:
        st.session_state.store = {}

    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None

    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        with st.spinner("Loading and processing PDFs..."):
            documents = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_pdf_path = temp_file.name

                loader = PyPDFLoader(temp_pdf_path)
                docs = loader.load()
                documents.extend(docs)

                os.remove(temp_pdf_path)

            # Split documents for embedding
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            splits = text_splitter.split_documents(documents)

            # Create vectorstore and retriever
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever()

            # Save to session state
            st.session_state.retriever = retriever
            st.session_state.vectorstore = vectorstore
        st.success("PDFs processed and indexed successfully!")

    if st.session_state.retriever is None:
        st.info("Please upload PDF files to start chatting with their content.")
    else:
        retriever = st.session_state.retriever

        # Standalone question generation prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # Q&A generation prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise.\n\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your question:")
        if user_input:
            session_history = get_session_history(session_id)
            try:
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}}
                )
                if response and "answer" in response:
                    st.markdown(f"**Assistant:** {response['answer']}")
                else:
                    st.warning("No valid answer returned. Please try again.")
                # Chat history is NOT displayed here to hide it from UI
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
else:
    st.warning("Please enter your Groq API key.")
