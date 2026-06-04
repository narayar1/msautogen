import asyncio
import uuid

from autogen_agentchat.teams import (
    DiGraphBuilder,
    GraphFlow
)

from autogen_ext.models.openai import (
    OpenAIChatCompletionClient
)

from autogen_core.models import (
    ModelInfo
)

from config import (
    GOOGLE_API_KEY,
    MODEL_NAME
)

from app.agents import (
    create_writer,
    create_reviewer
)

from observability.telemetry import (
    initialize_tracer
)

from observability.langfuse_client import (
    langfuse
)

from observability.event_processor import (
    process_event
)

tracer = initialize_tracer()

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

writer = create_writer(
    model_client
)

reviewer = create_reviewer(
    model_client
)

builder = DiGraphBuilder()

builder.add_node(writer)
builder.add_node(reviewer)

builder.add_edge(
    writer,
    reviewer
)

graph = builder.build()

team = GraphFlow(
    [writer, reviewer],
    graph
)

async def main():

    task = "Write a poem about India"

    trace_id = str(
        uuid.uuid4()
    )

    trace = langfuse.trace(
        name="poem-workflow",
        input=task
    )

    stream = team.run_stream(
        task=task
    )

    with tracer.start_as_current_span(
        "workflow"
    ):

        async for event in stream:

            print(event)

            process_event(
                event=event,
                trace_id=trace_id,
                workflow_name="poem-workflow"
            )

    langfuse.flush()


if __name__ == "__main__":
    asyncio.run(main())