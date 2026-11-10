import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4.1")

vector_store = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_template(
    "Answer the question based only on the context below: {context} \n Question: {question}\n Provide a detailed answer:"
)


def _format_content(documents):
    """Formats the content of the documents into a single string."""
    return "\n\n".join([doc.page_content for doc in documents])


def retrieval_chain_without_lcel(query: str):
    docs = retriever.invoke(query)
    context = _format_content(docs)
    messages = prompt.format_messages(context=context, question=query)
    response = llm.invoke(messages)
    return response.content


if __name__ == "__main__":
    question = "What is Pinecone in machine learning?"
    print("Implementation 1: Retrieval Chain without LCEL")
    result = retrieval_chain_without_lcel(question)
    print(result)
