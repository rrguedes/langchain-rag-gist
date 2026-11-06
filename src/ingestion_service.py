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

    loader = UnstructuredLoader(file_path=file_path)
    document = loader.load()
    print("Terminou")
