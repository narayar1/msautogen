from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import CancellationToken
from autogen_core.models import ModelInfo
from dotenv import load_dotenv
import os
import asyncio
from autogen_agentchat.ui import Console
from langfuse import Langfuse


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
model_name = os.environ.get("GEMINI_MODEL")
langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
langfuse_base_url = os.getenv("LANGFUSE_BASE_URL")

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

langfuse = Langfuse(
    public_key=langfuse_public_key,
    secret_key=langfuse_secret_key,
    host=langfuse_base_url
)

def temp_convertor_tool(temp: str) -> str:
    if temp.endswith("C"):
        celsius = float(temp[:-1])
        fahrenheit = (celsius * 9/5) + 32
        return f"{fahrenheit}F"
    elif temp.endswith("F"):
        fahrenheit = float(temp[:-1])
        celsius = (fahrenheit - 32) * 5/9
        return f"{celsius}C"
    else:
        raise ValueError("Temperature must end with 'C' or 'F'")


trace = langfuse.trace(
    name="temp_convert",
    user_id="user-001"
)

trace = langfuse.trace(
    name="temp_convert",
    user_id="user-001"
)
#Generation Span


agent = AssistantAgent(
    name="temp_agent",
    model_client=model_client,
    system_message=(
        "You are an agent who when provided the temperature in centigrade will "
        "convert to Fahrenheit and when provided the temperature in Fahrenheit will convert to centigrade"
),
    model_client_stream=True,
    tools=[temp_convertor_tool],   # <-- FIXED placement
)




async def agent_task():
    message = TextMessage(content="What is 100F in Celsius?", source ="user")
    generation = trace.generation(
    name="temp_convert",
    model="llama3",
    input=message
)
    generation = trace.generation(
    name="temp_convert",
    model="llama3",
    input=message
)

    response = await agent.on_messages([message], cancellation_token = CancellationToken())
    print("Agent response:", response.chat_message.content)
    generation.end(
    output=response.chat_message.content)
    print(f"\nTrace ID: {trace.id}")
    print(f"\n Output: {response.chat_message.content}")
    

    
if __name__ == "__main__":   
    asyncio.run(agent_task())
