import os
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL_NAME = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

vector_store_cache = {}

def get_ollama_embeddings():
    """Initializes and returns OllamaEmbeddings"""
    return OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)

def create_vector_store_from_text(doc_id: str, text_content: str):
    """
    Creates or updates an in-memory FAISS vector store for the given text 
    """
    print(f"Processing text for document ID: {doc_id} to create vector store... ")
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200, 
            length_function=len
        )
        chunks = text_splitter.split_text(text_content)
        if not chunks:
            print("Text content resulted in no chunks.")
            return False
        
        print(f"Split text into {len(chunks)} chunks.")
        
        embeddings = get_ollama_embeddings()
        print("Generating embeddings and creating FAISS vector store...")
        
        # FAISS.from_texts can take some time for many chunks
        vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
        vector_store_cache[doc_id] = vector_store
        
        print(f"Vector store created and cached for doc_id: {doc_id}")
        return True
    except Exception as e:
        print(f"Error creating vector store for doc_id {doc_id}: {e}")
        import traceback
        traceback.print_exc()
        if doc_id in vector_store_cache:
            del vector_store_cache[doc_id] # Clean up on error
        return False
        
def get_retriever_for_doc(doc_id: str, k_results: int = 3):
    """
    Returns a retriever for the specified doc_id from the cache.
    'k_results' is the number of relevant chunks to retrieve.
    """
    if doc_id in vector_store_cache:
        return vector_store_cache[doc_id].as_retriever(search_kwargs={"k": k_results})
    else:
        print(f"No vector store found for doc_id: {doc_id}")
        return None
    
# Example Test
if __name__ == '__main__':
    from dotenv import load_dotenv
    import asyncio
    load_dotenv() # Make sure .env is in backend directory etc.

    # First, ensure your chosen embedding model is running in Ollama
    # E.g., ollama run nomic-embed-text (it will download if not present, then exit. Server keeps it loaded)
    # Or `ollama pull nomic-embed-text`
    print(f"Attempting to use Ollama embedding model: {EMBEDDING_MODEL_NAME}")
    test_doc_id = "sample_document_1"
    sample_text_content = (
        "The Llama 3 series of large language models (LLMs) was released by Meta AI in April 2024. "
        "It includes models with 8 billion and 70 billion parameters. "
        "These models were trained on a significantly larger dataset compared to their predecessors. "
        "Llama 3 models are designed to be more helpful and safer, with improved reasoning abilities. "
        "The 8B parameter model, in particular, has shown strong performance on various benchmarks, "
        "often outperforming larger models from previous generations. "
        "They support a context length of 8,192 tokens, with some variants extending this. "
        "Meta has emphasized open access for these models, making them available for research and commercial use under a specific license."
    )

    success = create_vector_store_from_text(test_doc_id, sample_text_content)
    if success:
        print(f"\nVector store for '{test_doc_id}' created.")
        retriever = get_retriever_for_doc(test_doc_id)
        if retriever:
            query = "What are the key features of Llama 3 models?"
            print(f"\nTesting retriever with query: '{query}'")
            # LangChain's retriever is async by default for some methods.
            # For a simple sync test of underlying functionality:
            try:
                # For direct FAISS retriever (non-async by default in base retriever)
                # relevant_chunks = retriever.get_relevant_documents(query)
                # For async version if retriever is async enabled:
                async def test_retriever():
                    return await retriever.ainvoke(query)
                relevant_chunks = asyncio.run(test_retriever())

                print(f"\nRetrieved {len(relevant_chunks)} chunks:")
                for i, chunk in enumerate(relevant_chunks):
                    print(f"--- Chunk {i+1} ---")
                    print(chunk.page_content)
                    print("-" * 20)
            except Exception as e:
                print(f"Error during retriever test: {e}")
    else:
        print(f"\nFailed to create vector store for '{test_doc_id}'.")