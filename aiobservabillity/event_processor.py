import uuid
from datetime import datetime

from llmops_observability.models import AgentExecution
from .exporters.sqlite_exporter import export_execution


def process_event(
    event,
    trace_id,
    workflow_name,
    span_id = str(uuid.uuid4()),
    session_id=None,
    model_name=None,
    prompt=None,
    start_time=None
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

        prompt_tokens = getattr(
            usage,
            "prompt_tokens",
            0
        )

        completion_tokens = getattr(
            usage,
            "completion_tokens",
            0
        )

    total_tokens = (
        prompt_tokens +
        completion_tokens
    )

    INPUT_COST_PER_1K = 0.00015
    OUTPUT_COST_PER_1K = 0.00060

    cost_usd = (
    (prompt_tokens / 1000) * INPUT_COST_PER_1K
    + (completion_tokens / 1000) * OUTPUT_COST_PER_1K)
    response = str(
        getattr(
            event,
            "content",
            ""
        )
    )

    end_time = datetime.now()

    latency = 0.0

    if start_time:

        latency = (
            end_time -
            start_time
        ).total_seconds()

    record = AgentExecution(

        trace_id=trace_id,

        parent_trace_id = trace_id,

        span_id = span_id,

        session_id=session_id,

        workflow_name=workflow_name,

        agent_name=event.source,

        model_name=model_name,

        prompt=prompt,

        response=response,

        prompt_tokens=prompt_tokens,

        completion_tokens=completion_tokens,

        total_tokens=total_tokens,

        latency_seconds=latency,

        status="SUCCESS",

        start_time=start_time,

        end_time=end_time,

        timestamp=end_time,

        cost_usd=round(cost_usd, 6)
    )

    export_execution(record)

    print(
        f"{event.source}"
        f" | tokens={total_tokens}"
        f" | latency={latency:.2f}s"
    )