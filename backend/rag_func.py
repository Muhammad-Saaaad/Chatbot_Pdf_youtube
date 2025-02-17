import os
from dotenv import load_dotenv

from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import YoutubeLoader
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from uuid import uuid4

load_dotenv()
google_embeddings = GoogleGenerativeAIEmbeddings(google_api_key=os.getenv("GEMINI_API_KEY"), model="models/text-embedding-004")


client = QdrantClient(":memory:")
vector_store_memory = None

def create_vector_store():

    client.create_collection(
        collection_name="pdf_yt",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )
    return client # collection created sucessfully

def load_vector_store(client):

    collection = client.get_collections().collections # obvesly the collection shall only be excatly one
    global vector_store_memory
    print("______________________________",collection)
    if not collection:
        client = create_vector_store()
        vector_store = QdrantVectorStore(
            client=client,
            collection_name="pdf_yt",
            embedding=google_embeddings
        )
        vector_store_memory = vector_store
    else:
        print("re-load")
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection[0].name, # getting the name of the collection
            embedding=google_embeddings
        )
        vector_store_memory = vector_store
    return vector_store

def insert_links(links):

    global client
    collections = client.get_collections().collections

    if not collections:
        client = create_vector_store()
    vector_store = load_vector_store(client)
    splitter_links = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)

    try:
        for link in links:
            yt_loader = YoutubeLoader.from_youtube_url(link)
            loaded_data = yt_loader.load()
            chunks_links = splitter_links.split_documents(loaded_data)
            uids_links = [str(uuid4()) for _ in range(len(chunks_links))]
            ids = vector_store.add_documents(documents=chunks_links, ids=uids_links)
    except Exception as e :
        return {"response": e}
    

def insert_pdf(file_text): # here i am inserting pdf
    """
    Inserts a PDF file's text into a vector store after splitting it into chunks.
    Args:
        file_text (str): The text content of the PDF file to be inserted.
    Returns:
        None
    The function performs the following steps:
    1. Wraps the PDF text content into a Document object with metadata indicating the source as a PDF file.
    2. Splits the document into smaller chunks using RecursiveCharacterTextSplitter.
    3. Checks if there are any existing collections in the vector store client.
    4. If no collections exist, it creates a new vector store client.
    5. Loads the vector store client.
    6. Generates unique IDs for each chunk of text.
    7. Adds the chunks of text to the vector store with the generated IDs.
    8. Prints a confirmation message indicating that the vector store has been created.
    """

    text_doc = [Document(
        page_content=file_text, 
        metadata={
            "source":"pdf file"
        }
    )]
    
    splitter_text = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks_text = splitter_text.split_documents(text_doc)

    global client
    collections = client.get_collections().collections

    if not collections:
        client = create_vector_store()
    vector_store = load_vector_store(client)
    uids_text = [str(uuid4()) for _ in range(len(chunks_text))]

    uuid_text = vector_store.add_documents(documents=chunks_text, ids=uids_text)
    print("Vector Store Created")

def retrieved_data(query):
    global vector_store_memory
    if vector_store_memory is None:
        vector_store = load_vector_store(client)
    else:
        vector_store = vector_store_memory
    results = vector_store.similarity_search(query, k=3)
    return results