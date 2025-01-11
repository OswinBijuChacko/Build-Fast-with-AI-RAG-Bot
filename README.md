# Build-Fast-with-AI-RAG-Bot
This project implements a Retrieval-Augmented Generation (RAG) pipeline for building an intelligent chatbot. The bot uses FAISS for document retrieval, HuggingFace embeddings for vectorization, and Google Generative AI (Gemini 2.0) for generating responses. The project also provides functionality to inspect the embeddings used in the vector store.

#Features
Embedding generation using HuggingFace all-MiniLM-L6-v2
FAISS-based vector store for efficient similarity search
CSV-based knowledge base

Google Generative AI for natural language responses (optional)

Functionality to inspect embeddings

Resources Used

Libraries and Frameworks

Google Generative AI (Gemini 2.0)

Used for generating responses.

API Key: Configured via genai.configure.

LangChain

For creating and managing documents, retrievers, and vector stores.

Modules used:

langchain.schema.Document

langchain_community.vectorstores.FAISS

langchain_community.embeddings.HuggingFaceEmbeddings

HuggingFace Embeddings

Model: all-MiniLM-L6-v2

Used for embedding document content into vectors.

FAISS

Used as a vector store for fast similarity-based retrieval.

Pandas

Used to load and manipulate the CSV-based knowledge base.

Tools and Platforms

Python

The programming language used for the implementation.

ChatGPT

Used for assistance in debugging and refining the implementation.

Support included:

Optimizing embedding and FAISS integration.

Removing unnecessary chatbot-related functions.

Structuring the project.

File Structure

main.py: The primary Python script containing the implementation.

knowledge_base.csv: The CSV file serving as the knowledge base for the chatbot.

README.md: Documentation for the project.

Setup and Usage

Clone the repository.

Install required dependencies:

pip install pandas langchain faiss-cpu huggingface-hub google-generativeai

Add your Google Generative AI API key:

genai.configure(api_key="YOUR_API_KEY")

Place your knowledge base as a CSV file (knowledge_base.csv) in the project directory. Ensure the file contains the following columns:

Question

Answer

Run the script to print embeddings:

python main.py

Acknowledgements

ChatGPT: For providing insights and code optimization.

HuggingFace: For their pre-trained embedding models.

LangChain: For simplifying vector store and document management.

Google Generative AI: For enabling advanced language generation.
