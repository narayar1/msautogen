import uuid
from datetime import datetime

trace_id = str(uuid.uuid4())
session_id = str(uuid.uuid4())

workflow_name = "otel-langfuse-workflow"

start_time = datetime.now()


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

from llmops_observability.telemetry import (
    initialize_tracer
)

from llmops_observability.exporters.langfuse_exporter import langfuse

from llmops_observability.event_processor import (
    process_event
)

tracer = initialize_tracer(
    service_name="llmopsai"
)

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
        name="otel-langfuse-workflow",
        input=task
    )

    stream = team.run_stream(
        task=task
    )

    with tracer.start_as_current_span(
        "workflow"
    ):

        async for event in stream:

            process_event(
        event=event,
        trace_id=trace_id,
        workflow_name=workflow_name,
        session_id=session_id,
        model_name="gpt-4o-mini",
        prompt=task,
        start_time=start_time
    )

    langfuse.flush()


if __name__ == "__main__":
    asyncio.run(main())