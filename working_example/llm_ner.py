from ast import literal_eval

from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain
from langchain.chains.base import Chain
from langchain.llms import OpenAI

try:
    with open('api.key.txt', 'r') as file:
        api_key = file.read().strip()
except FileNotFoundError:
    print("Please create a file called 'api_key.txt' in the same directory as this file and paste your")
    raise FileNotFoundError


def get_ner_recognition(schema: dict = {}) -> Chain:
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", openai_api_key=api_key)
    chain = create_extraction_chain(schema, llm)

    return chain


def get_extraction_schema(required_information: str) -> dict:
    example_schema = {
        "properties": {
            "person_full_name": {"type": "string"},
            "person_height": {"type": "integer"},
        },
        "required": ["person_full_name", "person_height"],
    }
    generation_task = f"""Generate a python dictionary object that contains schema for named entities as per example:
    {example_schema}

    using the following information requirements:
    {required_information}

    Output:
    """
    llm = OpenAI(temperature=0.1, openai_api_key=api_key)
    return literal_eval(
        llm(
            generation_task
        ).strip()
    )
