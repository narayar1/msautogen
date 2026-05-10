from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.memory import ListMemory, MemoryContent,MemoryMimeType
from autogen_core.models import ModelInfo
import re
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
import asyncio
import os
from dotenv import load_dotenv

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL")


# ============================================================
# MODEL CLIENT
# ============================================================
model_client = OpenAIChatCompletionClient(
    model=MODEL_NAME,
    api_key=GOOGLE_API_KEY,
    api_type="google",
    temperature=0.2,
    parallel_tool_calls=False,   # IMPORTANT FOR GEMINI
    model_info=ModelInfo(
        vision=True,
        function_calling=True,
        json_output=False,
        structured_output=False,
        family="unknown",
        multiple_system_messages=True
    )
)

async def mem_manage():    
    await user_memory.add(MemoryContent(content=' weather should be in memory unit', mime_type=MemoryMimeType.TEXT))
    await user_memory.add(MemoryContent(content='Meal Recipe should be vegan', mime_type=MemoryMimeType.TEXT))
      
async def get_weather(city:str, units:str = "imperial") -> str:
    if units == "imperial":
        return f"The weather in {city} is 73 degrees  an Sunny"
    elif units == "metric":
        return f"The weather in {city} is 23 degrees  an Sunny"
    else:
        return "Invalid units. Please choose 'imperial' or 'metric'."

user_memory = ListMemory()
assistant_agent = AssistantAgent(
    name = "assistant_agent",
    description = "An assistant agent that provides weather information and meal recipes based on user preferences.",
    model_client=model_client,
    system_message="You are an assistant that provides weather information and also use tools whenver necessary",
    memory = [user_memory],
    tools = [get_weather]
)
stream = assistant_agent.run_stream(task = "Give me a receipe for soup dont ask further questions?")

asyncio.run(mem_manage())

async def main():
    await  Console(stream)
    await assistant_agent.model_context.get_messages()

if __name__ == "__main__":
    
    asyncio.run(main())