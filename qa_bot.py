from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


def load_qa_vector_index(api_key: str) -> Chroma:
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return Chroma(persist_directory='./chroma_storage/', embedding_function=embeddings)


def init() -> ConversationalRetrievalChain:
    try:
        with open('api.key.txt', 'r') as file:
            api_key = file.read().strip()
    except FileNotFoundError:
        print("Please create a file called 'api_key.txt' in the same directory as this file and paste your")
        raise FileNotFoundError

    vectorstore = load_qa_vector_index(api_key=api_key)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key='question', output_key='answer'
    )
    memory.chat_memory.add_user_message(
        "Hallo, ich bin der Chatbot der Raiffeisenbank. "
        "Ich gebe Ihnen kurze und relevante Antworten auf Ihre Fragen. "
        "Wie kann ich ihnen helfen?"
    )

    return ConversationalRetrievalChain.from_llm(
        OpenAI(temperature=0.1, openai_api_key=api_key),
        vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True
    )


if __name__ == '__main__':
    qa_chain = init()
    while qa_chain:
        # query = "KANN ICH MIT RAIPAY AUCH IM AUSLAND ZAHLEN?"
        query = input("Frage: ")
        if query == "exit":
            qa_chain = None
            break
        else:
            result = qa_chain({"question": query})
            print(result)
