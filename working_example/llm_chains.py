from langchain import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI


def get_intent_recognition_chain(api_key: str):
    intent_recognition_template = """You are a smart intent recognizing AI.

    Based on human input choose and output only the name of one of these intents:
    opening bank account: when customer explicitly asks for blocking opening a new bank account
    asking questions about banking: when customer asks questions about RAIPAY banking application or atm related questions
    blocking bank card: customer asks to block his bank card
    chitchat: chitchat about anything else

    Current conversation:
    {chat_history}
    Human: {question}
    AI:"""

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key='question', output_key='answer'
    )

    prompt_default_template = intent_recognition_template

    prompt_default = PromptTemplate(
        template=prompt_default_template, input_variables=['chat_history', 'question']
    )
    return ConversationChain(
        llm=OpenAI(openai_api_key=api_key),
        prompt=prompt_default,
        input_key='question',
        output_key='answer',
        memory=memory
    )


def get_customer_info_query_chain(api_key: str, required_information: str):
    polite_support_template = """You are a german speaking customer support agent helping people block their cards or open new accounts.

    Based on definition of information needed
    ask human in german for information they have not provided yet, 
    never mention calling external services or apis,
    only confirm to proceed once all information is provided,
    and never ask for the same information twice.

    Current conversation:
    {chat_history}

    Required information:
    """ + required_information + """
    Human: {question}
    AI:"""

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key='question',
        output_key='answer'
    )
    prompt = PromptTemplate(
        template=polite_support_template, input_variables=['chat_history', 'question']
    )
    return ConversationChain(
        llm=OpenAI(openai_api_key=api_key),
        prompt=prompt,
        input_key='question',
        output_key='answer',
        memory=memory
    )


def get_chitchat_chain(api_key: str):
    polite_support_template = """You are a german speaking customer support agent helping people answering their banking questions.

    Be polite, but try to keep answers short and related to banking domain.

    Current conversation:
    {chat_history}
    
    Human: {question}
    AI:"""

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key='question',
        output_key='answer'
    )
    prompt = PromptTemplate(
        template=polite_support_template, input_variables=['chat_history', 'question']
    )
    return ConversationChain(
        llm=OpenAI(openai_api_key=api_key),
        prompt=prompt,
        input_key='question',
        output_key='answer',
        memory=memory
    )
