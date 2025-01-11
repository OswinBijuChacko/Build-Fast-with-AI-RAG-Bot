import os
import google.generativeai as genai
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st

# Configure the Google Generative AI API key
genai.configure(api_key="AIzaSyAlP1dXEG4MEWXLlEVA1F7aCumVo08zwD8")

# Create the model with the generation configuration
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-2.0-flash-exp",
  generation_config=generation_config,
)

# Load the CSV file into a DataFrame
df = pd.read_csv("knowledge_base.csv")

# Convert the DataFrame into a list of Documents
documents = [Document(page_content=row['Answer']) for _, row in df.iterrows()]

# Extract questions for reference
questions = df['Question'].tolist()

# Initialize the HuggingFace embedding function
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Extract raw text from documents
texts = [doc.page_content for doc in documents]

# Create FAISS vector store directly from texts using the embedding function
vector_store = FAISS.from_texts(texts=texts, embedding=embedding_function)

# Set up a retriever using FAISS
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Define the prompt for generation
def build_prompt(question, context):
    return f"""
    You are an intelligent assistant helping answer questions based on the provided documents.

    Question: {question}
    Context:
    {context}

    Answer:
    """

# Function to generate a response using the generative model
def generate_response(prompt):
    # Start the chat session
    chat_session = model.start_chat(history=[])
    
    # Send the message with the prompt
    response = chat_session.send_message(prompt)
    
    return response.text  # Return the generated text

# Define a function to query the bot
def ask_bot(question):
    # Retrieve relevant documents
    retrieved_docs = retriever.get_relevant_documents(question)

    # Combine retrieved document content into a context string
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # Format the prompt with the question and context
    formatted_prompt = build_prompt(question, context)

    # Generate a response using the generative model
    response = generate_response(formatted_prompt)
    return response

# Streamlit app
st.title("RAG Chatbot with Streamlit")

# Input box for the user's question
question = st.text_input("Ask a question:", placeholder="Type your question here...")

if st.button("Get Answer"):
    if question:
        with st.spinner("Fetching answer..."):
            answer = ask_bot(question)
        st.write(f"**Q:** {question}")
        st.write(f"**A:** {answer}")
    else:
        st.warning("Please enter a question.")
