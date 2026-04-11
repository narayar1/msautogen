from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from autogen_core.models import ModelInfo
from autogen_core import CancellationToken
from dotenv import load_dotenv
import os
import asyncio

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
    message = TextMessage(content="I had like o go to London", source ="user")
    response = await agent.on_messages([message], cancellation_token = CancellationToken())
    print("Agent response:", response.chat_message.content)

if __name__ == "__main__":
    asyncio.run(agent_task())



