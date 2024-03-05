import os
from getpass import getpass

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

os.environ["ANTHROPIC_API_KEY"] = getpass()

chat = ChatAnthropic(temperature=0, model_name="claude-3-opus-20240229")

system = (
    "You are a helpful assistant that translate jokes {input_language} to {output_language} in natural and funny way."
)
human = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | chat
chain.invoke(
    {
        "input_language": "English",
        "output_language": "Japanese",
        "text": "A blind man walks into a fish market and says, 'Hello ladies.'",
    }
)