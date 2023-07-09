from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from llm_ner import get_ner_recognition, get_extraction_schema
from llm_chains import get_intent_recognition_chain, get_customer_info_query_chain, get_chitchat_chain
from llm_indices import load_qa_vector_index, load_open_block_vector_index, get_qa_response


def customer_requires_changes_flow(qa_chain: Chroma, query: str):
    action_definition_search = qa_chain({"question": query})
    action_requirements = action_definition_search["source_documents"][0].page_content

    extracted_information = {}
    extraction_schema = get_extraction_schema(action_requirements)

    ner_extractor = get_ner_recognition(extraction_schema)
    customer_agent = get_customer_info_query_chain(api_key, action_requirements)
    while not len([value for value in extracted_information.values() if value]) == 4:
        print(
            customer_agent.run(query)
        )
        query = input()
        extracted_information |= ner_extractor.run(query)[0]

    print(f'{intent.title()} with customer information: ', extracted_information)


def init(api_key: str) -> tuple[ConversationalRetrievalChain, ConversationalRetrievalChain]:
    vectorstore = load_qa_vector_index(api_key=api_key)
    vectorstore_block_or_open = load_open_block_vector_index(api_key=api_key)
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

    return (
        ConversationalRetrievalChain.from_llm(
            OpenAI(temperature=0.1, openai_api_key=api_key),
            vectorstore_block_or_open.as_retriever(search_kwargs={"k": 2}),
            memory=memory,
            return_source_documents=True
        ),
        ConversationalRetrievalChain.from_llm(
            OpenAI(temperature=0.1, openai_api_key=api_key),
            vectorstore.as_retriever(),
            memory=memory,
            return_source_documents=True
        )
    )


if __name__ == '__main__':
    try:
        with open('api.key.txt', 'r') as file:
            api_key = file.read().strip()
    except FileNotFoundError:
        print("Please create a file called 'api_key.txt' in the same directory as this file and paste your")
        raise FileNotFoundError

    welcome_msg = (
        "Hallo, ich bin der Chatbot der Raiffeisenbank. "
        "Ich gebe Ihnen kurze und relevante Antworten auf Ihre Fragen. "
    )
    print(welcome_msg)

    intent_recognizer = get_intent_recognition_chain(api_key=api_key)
    block_open_chain, qa_chain = init(api_key=api_key)
    chit_chatter = get_chitchat_chain(api_key=api_key)
    print("Wie kann ich ihnen helfen?")
    while qa_chain:
        query = input("Frage: ")
        if query == "exit":
            print("Auf Wiedersehen!")
            break
        else:
            intent = intent_recognizer.run({"question": query}).strip().lower()
            if intent == "opening bank account":
                customer_requires_changes_flow(block_open_chain, query)
            elif intent == "blocking bank card":
                customer_requires_changes_flow(block_open_chain, query)
            elif intent == "asking questions about banking":
                print(get_qa_response(qa_chain, query))
            else:
                print(chit_chatter(query)['answer'])
