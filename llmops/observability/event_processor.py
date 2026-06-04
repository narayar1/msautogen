import time
from datetime import datetime

from observability.models import AgentExecution
from storage.delta_logger import write_execution


def process_event(
    event,
    trace_id,
    workflow_name
):

    if not hasattr(event, "source"):
        return

    usage = getattr(
        event,
        "models_usage",
        None
    )

    prompt_tokens = 0
    completion_tokens = 0

    if usage:

        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens

    total_tokens = (
        prompt_tokens +
        completion_tokens
    )

    response = str(
        getattr(
            event,
            "content",
            ""
        )
    )

    record = AgentExecution(

        trace_id=trace_id,

        workflow_name=workflow_name,

        agent_name=event.source,

        prompt_tokens=prompt_tokens,

        completion_tokens=completion_tokens,

        total_tokens=total_tokens,

        response=response,

        latency_seconds=0.0,

        status="SUCCESS",

        timestamp=datetime.now()
    )

    write_execution(record)

    print(
        f"{event.source} "
        f"-> {total_tokens} tokens"
    )