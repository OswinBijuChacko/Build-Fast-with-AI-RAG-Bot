Libraries and Frameworks
Google Generative AI (Gemini 2.0): Used for generating responses through advanced language generation capabilities. The API key is configured via genai.configure.

LangChain: A powerful tool for creating and managing documents, retrievers, and vector stores. Key modules used include:

langchain.schema.Document
langchain_community.vectorstores.FAISS
langchain_community.embeddings.HuggingFaceEmbeddings

HuggingFace Embeddings: The model all-MiniLM-L6-v2 is used to embed document content into vectors, enabling efficient retrieval of information.
FAISS: A vector store used to facilitate fast similarity-based retrieval, improving the chatbot's performance in matching relevant information.
Pandas: Utilized for loading and manipulating the CSV-based knowledge base, making data handling straightforward.

Tools and Platforms
Python: The programming language used for the implementation of the project.
ChatGPT: Assisted in debugging, optimizing embedding and FAISS integration, removing unnecessary chatbot-related functions, and structuring the project.

File Structure
main.py: The primary Python script that contains the main implementation logic.

knowledge_base.csv: A CSV file used as the knowledge base for the chatbot, containing questions and answers.

README.md: Documentation for setting up and understanding the project.
Setup and Usage
Clone the repository.
Acknowledgements
ChatGPT: For providing insights and assisting in code optimization.
HuggingFace: For their pre-trained embedding models.
LangChain: For simplifying the management of vector stores and documents.
Google Generative AI: For enabling advanced language generation capabilities.
