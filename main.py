import os
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI

load_dotenv()
claudea_chat = ChatAnthropic(temperature=0.5, anthropic_api_key=os.environ["ANTHROPIC_API_KEY"], model_name="claude-3-opus-20240229")
openai_chat = ChatOpenAI(temperature=0.5, openai_api_key=os.environ["OPENAI_API_KEY"],model_name="gpt-4-turbo-preview")


system = (
    "Give the best questions to measure the two llm models. return in {output_language}"
)
human = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
text= "I'm comparing the two chat models. I'm trying to see which one is better. the questions needs to be something that can show the potential of the model, ideally something the answer would be measurable and doesn't require current information. return in Japanese"
chain = prompt | claudea_chat 
ans=chain.invoke(
    {
        "output_language": "Japanese",
        "text":text,
    }
)
print(ans)

system_message_prompt = SystemMessagePromptTemplate.from_template(system)
human_message_prompt = HumanMessagePromptTemplate.from_template(human)
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

# get a chat completion from the formatted messages
ans_openai=openai_chat.invoke(
    chat_prompt.format_prompt(
        output_language="Japanese", 
        text=text
    ).to_messages()
)
print("\n")
print("gpt",ans_openai)
print(chat_prompt.format_prompt(
        output_language="Japanese", 
        text=text
    ).to_messages())