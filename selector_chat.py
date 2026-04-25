import asyncio
from dotenv import load_dotenv
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.ui import Console

from autogen_ext.models.openai import OpenAIChatCompletionClient # type: ignore
from autogen_core.models import ModelInfo # type: ignore
from typing import Sequence
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage


# ================================
# LOAD ENV VARIABLES
# ================================
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL")


# ================================
# MODEL CLIENT
# ================================
model_client = OpenAIChatCompletionClient(
    model=MODEL_NAME,
    api_key=GOOGLE_API_KEY,
    api_type="google",
    temperature=0.7,
    model_info=ModelInfo(
        vision=True,
        function_calling=True,
        json_output=True,
        structured_output=True,
        family="unknown"
    )
)


# ================================
# TOOL FUNCTIONS
# ================================
def search_web_tool(query: str) -> str:
    """
    Searches historical Miami Heat player statistics.
    """
    print(f"\n[search_web_tool invoked with query]: {query}\n")

    if "2006-2007" in query or "2006-07" in query:
        return """
        Here are the total points scored by Miami Heat players in 2006-2007 season:
        Udonis Haslem : 844 points
        Dwayne Wade : 1397 points
        James Posey : 550 points
        """
    elif "2007-2008" in query or "2007-08" in query:
        return """
        The total rebounds for Dwayne Wade in 2007-2008 season is 214
        """
    elif "2008-2009" in query or "2008-09" in query:
        return """
        The total rebounds for Dwayne Wade in 2008-2009 season is 398
        """
    else:
        return "No data available for the query."


def percentage_change_tool(start: float, end: float) -> float:
    """
    Calculates percentage change between two numeric values.
    """
    print(f"\n[percentage_change_tool invoked with]: start={start}, end={end}\n")

    if start == 0:
        return 0.0
    return ((end - start) / abs(start)) * 100


# ================================
# AGENTS
# ================================

planning_agent = AssistantAgent(
    name="PlanningAgent",
    description="Plans tasks and delegates work to the right agents.",
    model_client=model_client,
    system_message="""
You are a Planning Agent.

Your responsibility is to break the user request into smaller subtasks
and assign them to the appropriate team members.

Available team members:
1. WebSearchAgent - searches for factual information
2. DataAnalystAgent - performs mathematical calculations

STRICT RULES:
- You do not execute any task yourself.
- You only delegate tasks.
- Always assign tasks in this exact format:

<agent> : <task>

- After assigning, wait for agents to complete.
- Once all agent responses are available, summarize the final answer.
- End the final answer with TERMINATE.
"""
)


web_search_agent = AssistantAgent(
    name="WebSearchAgent",
    description="Searches for historical sports information using search_web_tool.",
    model_client=model_client,
    tools=[search_web_tool],
    system_message="""
You are a Web Search Agent.

Your only responsibility is to search and retrieve factual information.

Rules:
- Use search_web_tool whenever information is needed.
- Only make one tool call at a time.
- Do not perform calculations.
- Return only the factual results found.
"""
)


data_analyst_agent = AssistantAgent(
    name="DataAnalystAgent",
    description="Performs percentage calculations using percentage_change_tool.",
    model_client=model_client,
    tools=[percentage_change_tool],
    system_message="""
You are a Data Analyst Agent.

Your job is to perform numerical calculations.

Rules:
- Use percentage_change_tool whenever a percentage change calculation is requested.
- If the required numbers are not available in conversation, ask for them.
- Return only the analytical result.
"""
)


# ================================
# TERMINATION CONDITIONS
# ================================
text_mention_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(20)
combined_termination = text_mention_termination | max_messages_termination


# ================================
# SELECTOR PROMPT
# ================================
selector_prompt = """
You are the speaker selection coordinator for a multi-agent team.

Available participants:
{roles}

Conversation history:
{history}

Select exactly one next speaker from {participants}.

Rules:
1. PlanningAgent must always speak first.
2. PlanningAgent delegates tasks to worker agents.
3. After a worker agent completes, choose the next required worker or PlanningAgent.
4. Once all worker results are available, PlanningAgent should summarize.
5. Only output one participant name.
"""


# ================================
# TEAM
# ================================

def selector_func(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
    if messages[-1].source != planning_agent.name:
        return planning_agent.name
    return None

def selector_team():
    team = SelectorGroupChat(
        participants=[planning_agent, web_search_agent, data_analyst_agent],
        model_client=model_client,
        termination_condition=combined_termination,
        selector_prompt=selector_prompt,
        allow_repeated_speaker=True,
        selector_func=selector_func
    )
    return team




# ================================
# TASK
# ================================
task = (
    "Who was the Miami Heat player with the most points in the 2006-2007 season "
    "and what was the percentage change in his total rebounds between the 2007-08 "
    "and 2008-09 seasons?"
)



# ================================
# MAIN
# ================================
async def main():
    team = selector_team()
    await Console(team.run_stream(task=task))


if __name__ == "__main__":
    asyncio.run(main())