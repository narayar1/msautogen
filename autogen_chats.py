from urllib import response

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from autogen_core.models import ModelInfo
from autogen_core import CancellationToken
from dotenv import load_dotenv
import os
import asyncio
import sqlite3

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
model_name = os.environ.get("GEMINI_MODEL")
print(model_name)

model_client = OpenAIChatCompletionClient(    
    model=model_name,
    api_key=GOOGLE_API_KEY,
    api_type = "google",
    model_info=ModelInfo(
        vision=True,
        function_calling=True,
        temperature=0.7,
        json_output=True,
        structured_output=True,
        family="unknown",

    )
)

agent = AssistantAgent(
    name= "airline_agent",
    model_client=model_client,
    system_message="You are a helpful assistantfor an airline. You give humorous answers",
    model_client_stream  = True
)


async def agent_task():
    message = TextMessage(content="I had like to go to London", source ="user")
    response = await agent.on_messages([message], cancellation_token = CancellationToken())
    print("Agent response:", response.chat_message.content)

if __name__ == "__main__":
    asyncio.run(agent_task())

if os.path.exists("tickets.db"):
    os.remove("tickets.db")

# Create Table and Database
conn = sqlite3.connect("tickets.db")
c = conn.cursor()
c.execute("create table cities (city_name text PRIMARY KEY," \
"round_trip_price REAL)")       
conn.commit()
conn.close()

def save_city_price(city_name,round_trip_price):
    conn = sqlite3.connect("tickets.db")
    c = conn.cursor()
    c.execute("REPLACE into cities (city_name, round_trip_price) values (?,?)", (city_name.lower(), round_trip_price))
    conn.commit()
    conn.close()

# save cities

save_city_price("London", 299)
save_city_price("Paris", 399)
save_city_price("Rome", 499)
save_city_price("Madrid", 550)
save_city_price("Barcelona", 599)
save_city_price("Berlin", 525)

def get_city_price(city_name:str) -> float|None:
    """ Get the roundtrip ticket price to travel to the city"""
    conn = sqlite3.connect("tickets.db")
    c = conn.cursor()
    c.execute("SELECT round_trip_price from cities where city_name = ?", (city_name.lower(),))
    result = c.fetchone()
    conn.close()
    if result:
        return result[0]
    else:
        return None
    
get_city_price("London")

async def smartagents():
    smart_agent = AssistantAgent(
    name= "smart_airline_agent",
    model_client=model_client,
    system_message="You are a helpful assistantfor an airline." \
    " You give short humorous answers including the price of " \
    "roundtrip ticket",
    model_client_stream  = True,
    tools = [get_city_price],
    reflect_on_tool_use = True
    )
    message = TextMessage(content="I had like to go to London", source ="user")
    response = await smart_agent.on_messages([message],
    cancellation_token= CancellationToken())
    
    for message in response.inner_messages:
        print(message.content)
    print(response.chat_message.content)

asyncio.run(smartagents())










