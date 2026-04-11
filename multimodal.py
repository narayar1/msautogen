from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage, MultiModalMessage
from autogen_core import Image as AGImage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from autogen_core.models import ModelInfo
from autogen_core import CancellationToken
from dotenv import load_dotenv
import os
import asyncio
from pydantic import BaseModel, Field
from typing import Literal
from IPython.display import display, Markdown
from PIL import Image 
from io import BytesIO
import requests
import asyncio
import textwrap

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
model_name = os.environ.get("GEMINI_MODEL")


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


url = "https://edwarddonner.com/wp-content/uploads/2024/10/from-software-engineer-to-AI-DS.jpeg"

pil_image = Image.open(BytesIO(requests.get(url).content))
img = AGImage(pil_image)

multi_modal_message = MultiModalMessage(content=["Describe the content of this image in detail", img], source = "user")

describer = AssistantAgent(
    name = "description_agent",
    model_client = model_client,
    system_message = "You are good at describing images"
)

async def description_task():
    response = await describer.on_messages([multi_modal_message],
    cancellation_token = CancellationToken())
    reply = response.chat_message.content
    print(Markdown(reply))
    print(reply)

asyncio.run(description_task())

#Structured output 

class ImageDescription(BaseModel):
    scene: str = Field(description="Describe the overall scene of the image")
    message: str = Field(description = "The point that the image is trying to convey")
    style: str  = Field(description = "The artistic style of the image")
    orientation: Literal["portrait","landscape","square"] = Field(description = "The orientation of the image")
    
imagedescriber = AssistantAgent(
    name = "description_agent",
    model_client = model_client,
    system_message = "You are good at describing images in detail",
    output_content_type = ImageDescription
)

async def format_description_task():
    response = await imagedescriber.on_messages([multi_modal_message],
    cancellation_token = CancellationToken())
    reply = response.chat_message.content
    print(reply)
    print(f"Scene:\n{textwrap.fill(reply.scene)}\n\n")
    print(f"Message:\n{textwrap.fill(reply.message)}\n\n")
    print(f"Style:\n{textwrap.fill(reply.style)}\n\n")
    print(f"Orientation:\n{textwrap.fill(reply.orientation)}\n\n")


asyncio.run(format_description_task())

