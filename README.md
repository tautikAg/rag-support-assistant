# <Company-Name> Support Assistant

## Overview

<Company-Name> Support Assistant is an AI-powered chatbot designed to provide accurate and helpful responses to user queries about <Company-Name>. The assistant follows a structured approach to analyze, reason, and formulate responses based on the provided help center articles and documentation updates.

## Key Components

### 1. Document Embedding and Retrieval
The assistant uses a document embedding and retrieval pipeline to manage and query the company's documentation. The documents are stored in a ChromaDB collection, and embeddings are generated using OpenAI's embedding model.

### 2. Document Management
Documents are added to the collection from a JSON file (`company.json`). The documents are tokenized and chunked if they exceed a certain token limit to ensure efficient storage and retrieval.

### 3. Query Processing
When a user submits a query, the assistant retrieves relevant documents from the collection based on the query's context. The retrieved documents are used to generate a comprehensive response.

### 4. Response Generation
The assistant formulates responses by synthesizing information from the retrieved documents. It follows a structured approach to ensure the responses are accurate and helpful:
- **Analysis**: Analyzes the user's query to identify key issues or questions.
- **Reasoning**: Reviews the relevant documents to find exact answers or related contextual information.
- **Response Formulation**: Synthesizes the information to provide a clear and comprehensive answer.

### 5. Chat History Management
The assistant maintains a chat history to provide context for ongoing conversations. The chat history is stored in a JSON file (`msg.json`).

### 6. Knowledge Update
Users can update the knowledge base by adding new information to an `updated_docs.txt` file. This ensures the assistant stays up-to-date with the latest information.

## Usage

### Adding Documents to the Collection
Documents are added to the collection from a JSON file. The documents are tokenized and chunked if necessary to fit within the token limit.

### Querying the Assistant
Users can interact with the assistant through a chat interface. The assistant retrieves relevant documents based on the user's query and generates a response.

### Updating Knowledge
Users can update the knowledge base by adding new information to the `updated_docs.txt` file.

## Conclusion

<Company-Name> Support Assistant leverages advanced AI and document retrieval techniques to provide accurate and helpful responses to user queries. By maintaining an up-to-date knowledge base and following a structured approach to response generation, the assistant ensures users receive the best possible support.

### How `add_documents_to_collection` Works

The `add_documents_to_collection` function is responsible for adding documents to the ChromaDB collection. Here's a detailed explanation of the process:

1. **Clearing the Collection**: The function starts by clearing the existing collection to ensure no duplicate entries. This is done using `client.delete_collection("company_docs")`.

2. **Creating the Collection**: It then creates or retrieves the collection using `client.get_or_create_collection("company_docs", embedding_function=openai_ef)`.

3. **Loading Documents**: The documents are loaded from a JSON file (`company.json`). Each document contains markdown content.

4. **Tokenization and Chunking**: 
    - The documents are tokenized using the `tiktoken` library, which encodes the text into tokens.
    - If a document exceeds the maximum token limit (4000 tokens), it is split into smaller chunks. The number of chunks and the size of each chunk are calculated to ensure efficient storage and retrieval.

5. **Adding Documents to the Collection**: 
    - For each chunk or document, the function adds it to the collection with a unique ID and metadata. The ID is based on the document title and part number (if chunked), and the metadata includes the document URL.

This process ensures that the documents are efficiently stored and can be quickly retrieved based on user queries.