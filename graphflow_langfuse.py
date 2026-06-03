import asyncio
import os
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo

from langfuse import Langfuse

# ============================================================
# CONFIG
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL")

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_BASE_URL")

# ============================================================
# LANGFUSE
# ============================================================

langfuse = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host=LANGFUSE_HOST
)

# ============================================================
# MODEL CLIENT
# ============================================================

model_client = OpenAIChatCompletionClient(
    model=MODEL_NAME,
    api_key=GOOGLE_API_KEY,
    api_type="google",
    temperature=0.2,
    parallel_tool_calls=False,
    model_info=ModelInfo(
        vision=True,
        function_calling=True,
        json_output=False,
        structured_output=False,
        family="unknown"
    )
)

# ============================================================
# AGENTS
# ============================================================

writer = AssistantAgent(
    name="writer",
    description="Writer agent",
    model_client=model_client,
    system_message=(
        "You are a writer. "
        "Write a concise and engaging poem."
    )
)

reviewer = AssistantAgent(
    name="reviewer",
    description="Reviewer agent",
    model_client=model_client,
    system_message=(
        "Review the poem and provide feedback."
    )
)

# ============================================================
# GRAPHFLOW
# ============================================================

builder = DiGraphBuilder()

builder.add_node(writer)
builder.add_node(reviewer)

builder.add_edge(writer, reviewer)

graph = builder.build()

team = GraphFlow(
    participants=[writer, reviewer],
    graph=graph
)

# ============================================================
# MAIN
# ============================================================

async def main():

    task = "Write a good poem about India"

    # ------------------------------------------------
    # ROOT TRACE
    # ------------------------------------------------

    trace = langfuse.trace(
        name="india-poem-workflow",
        user_id="user-001",
        input=task
    )

    writer_span = None
    reviewer_span = None

    writer_output = ""
    reviewer_output = ""

    writer_prompt_tokens = 0
    writer_completion_tokens = 0

    reviewer_prompt_tokens = 0
    reviewer_completion_tokens = 0

    stream = team.run_stream(task=task)

    async for event in stream:

        print("\n====================")
        print(event)
        print("====================")

        # -------------------------------------------
        # WRITER EVENT
        # -------------------------------------------

        if hasattr(event, "source") and event.source == "writer":

            if writer_span is None:

                writer_span = trace.span(
                    name="writer-agent",
                    input=task
                )

            writer_output = event.content

            if hasattr(event, "models_usage") and event.models_usage:

                writer_prompt_tokens += event.models_usage.prompt_tokens
                writer_completion_tokens += event.models_usage.completion_tokens

        # -------------------------------------------
        # REVIEWER EVENT
        # -------------------------------------------

        elif hasattr(event, "source") and event.source == "reviewer":

            if reviewer_span is None:

                reviewer_span = trace.span(
                    name="reviewer-agent",
                    input=writer_output
                )

            reviewer_output = event.content

            if hasattr(event, "models_usage") and event.models_usage:

                reviewer_prompt_tokens += event.models_usage.prompt_tokens
                reviewer_completion_tokens += event.models_usage.completion_tokens

    # ------------------------------------------------
    # END SPANS
    # ------------------------------------------------

    if writer_span:

        writer_span.end(
            output=writer_output,
            metadata={
                "prompt_tokens": writer_prompt_tokens,
                "completion_tokens": writer_completion_tokens,
                "total_tokens": (
                    writer_prompt_tokens
                    + writer_completion_tokens
                )
            }
        )

    if reviewer_span:

        reviewer_span.end(
            output=reviewer_output,
            metadata={
                "prompt_tokens": reviewer_prompt_tokens,
                "completion_tokens": reviewer_completion_tokens,
                "total_tokens": (
                    reviewer_prompt_tokens
                    + reviewer_completion_tokens
                )
            }
        )

    # ------------------------------------------------
    # TRACE COMPLETE
    # ------------------------------------------------

    trace.update(
        output={
            "poem": writer_output,
            "review": reviewer_output
        }
    )

    langfuse.flush()

    # ------------------------------------------------
    # TOKEN SUMMARY
    # ------------------------------------------------

    print("\n")
    print("========================================")
    print("TOKEN USAGE SUMMARY")
    print("========================================")

    print("\nWriter Agent")

    print(
        f"Prompt Tokens     : {writer_prompt_tokens}"
    )

    print(
        f"Completion Tokens : {writer_completion_tokens}"
    )

    print(
        f"Total Tokens      : "
        f"{writer_prompt_tokens + writer_completion_tokens}"
    )

    print("\nReviewer Agent")

    print(
        f"Prompt Tokens     : {reviewer_prompt_tokens}"
    )

    print(
        f"Completion Tokens : {reviewer_completion_tokens}"
    )

    print(
        f"Total Tokens      : "
        f"{reviewer_prompt_tokens + reviewer_completion_tokens}"
    )

    print("\nWorkflow Total")

    grand_total = (
        writer_prompt_tokens
        + writer_completion_tokens
        + reviewer_prompt_tokens
        + reviewer_completion_tokens
    )

    print(f"Total Tokens: {grand_total}")

    try:
        print(f"\nTrace ID: {trace.id}")
    except:
        pass


if __name__ == "__main__":
    asyncio.run(main())