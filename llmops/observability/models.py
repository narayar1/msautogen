from dataclasses import dataclass
from datetime import datetime


@dataclass
class AgentExecution:

    trace_id: str
    workflow_name: str
    agent_name: str

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    response: str

    latency_seconds: float

    status: str

    timestamp: datetime