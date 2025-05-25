# Conversational-RAG-with-PDF-Uploads

Conversational RAG with PDF Uploads and Chat History
This Streamlit application enables you to upload multiple PDF documents, index their contents using vector embeddings, and engage in conversational retrieval-augmented generation (RAG) based question-answering over the uploaded documents. The app leverages LangChain, Groq’s ChatGroq LLM, and Chroma vector store to provide context-aware answers.

Features
PDF Uploads: Upload multiple PDF files which are parsed and split into manageable text chunks.

Vector Embeddings: Text chunks are embedded using OpenAI embeddings and stored in a Chroma vector store for efficient semantic search.

Conversational RAG: Chat with the content using a retrieval-augmented generation approach, maintaining chat context internally for coherent, standalone question reformulation.

Session Support: Support for multiple chat sessions identified by a session ID.

Secure API Key Input: Input your Groq API key securely for LLM access.

Chat History Management: Chat history is stored internally to maintain context but is hidden from the UI for privacy.

