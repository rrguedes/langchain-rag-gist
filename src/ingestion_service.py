import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_unstructured import UnstructuredLoader

load_dotenv()

if __name__ == "__main__":
    PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
    print("ingesting...")

    file_path = (
        r"C:\Users\rafae\Documents\Projetos\langchain-rag-gist\assets\mediumblog1.txt"
    )

    # 1. Loading
    loader = UnstructuredLoader(file_path=file_path)
    document = loader.load()

    # 2. Splitting
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)

    # 3. Ingesting
    embeddings = OpenAIEmbeddings(openai_api_type=os.environ["OPENAI_API_KEY"])
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )

    print("Terminou")
