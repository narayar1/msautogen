import asyncio
import os
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.ui import Console

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo

from openai import RateLimitError


# ==========================================
# LOAD ENV
# ==========================================
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL")


# ==========================================
# GEMINI SAFE MODEL CLIENT WITH RETRY
# ==========================================
class RetryGeminiClient(OpenAIChatCompletionClient):
    async def create(self, *args, **kwargs):
        retries = 5
        delay = 12

        for attempt in range(retries):
            try:
                return await super().create(*args, **kwargs)
            except RateLimitError as e:
                print(f"\n[Gemini Rate Limit Hit...Retrying in {delay} sec]\n")
                await asyncio.sleep(delay)

        raise Exception("Gemini quota exhausted after retries.")


model_client = RetryGeminiClient(
    model=MODEL_NAME,
    api_key=GOOGLE_API_KEY,
    api_type="google",
    temperature=0.3,
    model_info=ModelInfo(
        vision=True,
        function_calling=True,
        json_output=True,
        structured_output=True,
        family="unknown"
    )
)


# ==========================================
# TOOLS
# ==========================================
def search_web_tool(query: str) -> str:
    """
    Searches historical Miami Heat player statistics.
    """
    print(f"\n[search_web_tool called] -> {query}\n")

    if "2006-2007" in query or "2006-07" in query:
        return """
        Here are the total points scored by Miami Heat players in 2006-2007 season:
        Udonis Haslem : 844 points
        Dwayne Wade : 1397 points
        James Posey : 550 points
        """

    elif "2007-2008" in query or "2007-08" in query:
        return "The total rebounds for Dwayne Wade in 2007-2008 season is 214"

    elif "2008-2009" in query or "2008-09" in query:
        return "The total rebounds for Dwayne Wade in 2008-2009 season is 398"

    return "No data available."


def percentage_change_tool(start: float, end: float) -> float:
    """
    Calculates percentage change between two values.
    """
    print(f"\n[percentage_change_tool called] start={start}, end={end}\n")

    if start == 0:
        return 0.0

    return ((end - start) / abs(start)) * 100


# ==========================================
# AGENTS
# ==========================================
planning_agent = AssistantAgent(
    name="PlanningAgent",
    description="Plans tasks and summarizes final answer.",
    model_client=model_client,
    system_message="""
You are a planning agent.


STRICT RULES:
- Break task into subtasks only.
- Never assume any factual numbers.
- Never perform calculations.
- Never include guessed values in delegated tasks.
- Delegate only information gathering or analysis requests.
- After all worker results are available, summarize.
- End with TERMINATE.

Assign tasks exactly in this format:
<agent> : <task>
"""
)

web_search_agent = AssistantAgent(
    name="WebSearchAgent",
    description="Searches for sports facts using search_web_tool.",
    model_client=model_client,
    tools=[search_web_tool],
    system_message="""
You are a web search agent.



STRICT RULES:
- You MUST use search_web_tool for every factual query.
- Never answer from your own memory.
- Never infer or estimate sports statistics.
- Only return the exact values obtained from the tool.
- If tool gives no data, say no data available.
"""
)

data_analyst_agent = AssistantAgent(
    name="DataAnalystAgent",
    description="Calculates percentages using percentage_change_tool.",
    model_client=model_client,
    tools=[percentage_change_tool],
    system_message="""
You are a data analyst agent.

Rules:
- Perform calculations only.
- Use percentage_change_tool when required.
- Return the final numeric result.
"""
)


# ==========================================
# TERMINATION
# ==========================================
termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(15)


# ==========================================
# DETERMINISTIC SELECTOR FUNCTION
# ==========================================
def custom_selector(messages) -> str:
    """
    Deterministic speaker routing.
    This avoids LLM speaker selection calls.
    """

    if len(messages) == 0:
        return "PlanningAgent"

    last_speaker = messages[-1].source

    # Planning agent speaks first
    if last_speaker == "PlanningAgent":
        planning_text = messages[-1].content.lower()

        if "2006-2007" in planning_text or "2006-07" in planning_text:
            return "WebSearchAgent"

        return "WebSearchAgent"

    # WebSearchAgent should be called multiple times before analyst
    if last_speaker == "WebSearchAgent":
        web_count = len([m for m in messages if m.source == "WebSearchAgent"])

        if web_count < 3:
            return "WebSearchAgent"
        else:
            return "DataAnalystAgent"

    # After analyst return to planner
    if last_speaker == "DataAnalystAgent":
        return "PlanningAgent"

    return "PlanningAgent"


# ==========================================
# TEAM
# ==========================================
team = SelectorGroupChat(
    participants=[planning_agent, web_search_agent, data_analyst_agent],
    model_client=model_client,
    termination_condition=termination,
    selector_func=custom_selector,
    allow_repeated_speaker=True
)


# ==========================================
# TASK
# ==========================================
task = (
    "Who was the Miami Heat player with the most points in the 2006-2007 season "
    "and what was the percentage change in his total rebounds between the 2007-08 "
    "and 2008-09 seasons?"
)


# ==========================================
# MAIN
# ==========================================
async def main():
    await Console(team.run_stream(task=task))


if __name__ == "__main__":
    asyncio.run(main())