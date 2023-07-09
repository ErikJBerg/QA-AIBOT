from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma


def load_qa_vector_index(api_key: str) -> Chroma:
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return Chroma(persist_directory='./chroma_storage/', embedding_function=embeddings)


def get_qa_response(qa_chain: Chroma, query: str):
    return qa_chain({"question": query})["answer"]


def load_open_block_vector_index(api_key: str) -> Chroma:
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return Chroma(persist_directory='./chroma_storage_intents/', embedding_function=embeddings)
