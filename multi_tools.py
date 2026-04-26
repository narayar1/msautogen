import asyncio
import os
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import CancellationToken
from autogen_core.models import ModelInfo

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

# -----------------------------
# Tools
# -----------------------------
def travel_tool(destination: str) -> str:
    return f"Travel plan created for {destination}. Suggested attractions and itinerary ready."

def hotel_tool(city: str) -> str:
    return f"Hotel booked successfully in {city}. Reservation confirmed."

# -----------------------------
# Agents
# -----------------------------
travel_agent = AssistantAgent(
    name="travel_agent",
    model_client=model_client,
    system_message="You help users plan travel destinations.",
    tools=[travel_tool],
)

hotel_agent = AssistantAgent(
    name="hotel_agent",
    model_client=model_client,
    system_message="You handle hotel bookings.",
    tools=[hotel_tool],
)

# -----------------------------
# Router Agent
# -----------------------------
router_agent = AssistantAgent(
    name="router",
    model_client=model_client,
    system_message=(
        "You are a router.\n"
        "If user talks about travel, trip, destination → return 'travel_agent'\n"
        "If user talks about hotel, stay, booking → return 'hotel_agent'\n"
        "If both are present → return 'travel_agent,hotel_agent'\n"
        "ONLY return agent names, nothing else."
    ),
)

# -----------------------------
# Main Loop (Interactive)
# -----------------------------
async def main():
    print("🌍 Travel Assistant (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        # Step 1: Route decision
        route_response = await router_agent.on_messages(
            [TextMessage(content=user_input, source="user")],
            cancellation_token=CancellationToken(),
        )

        route = route_response.chat_message.content.strip()
        selected_agents = [r.strip() for r in route.split(",")]

        print(f"\n🔀 Routed to: {selected_agents}")

        # Step 2: Execute agents
        for agent_name in selected_agents:
            if agent_name == "travel_agent":
                agent = travel_agent
            elif agent_name == "hotel_agent":
                agent = hotel_agent
            else:
                continue

            response = await agent.on_messages(
                [TextMessage(content=user_input, source="user")],
                cancellation_token=CancellationToken(),
            )

            print(f"\n🤖 {agent.name}: {response.chat_message.content}")

        print("\n" + "-"*50 + "\n")


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    asyncio.run(main())