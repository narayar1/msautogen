from flask import Flask, request, jsonify, render_template
from autogen_core.models import ModelInfo
import asyncio
from dotenv import load_dotenv
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

app = Flask(__name__)

# Load API key securely
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


@app.route("/")
def home():
    return render_template("index.html")

async def run_autogen(user_input: str):
    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message="You are an agent who provides humorous responses."
    )

    response = await assistant.run(task=user_input)

    return response.messages[-1].content


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")

    result = asyncio.run(run_autogen(user_input))

    return jsonify({"response": result})


if __name__ == "__main__":
    app.run()