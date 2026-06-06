from dataclasses import dataclass
from datetime import datetime


@dataclass
class AgentExecution:

    trace_id: str

    parent_trace_id: str = None
    
    span_id: str = None

    session_id: str = None

    workflow_name: str = None
    agent_name: str = None
    model_name: str = None

    prompt: str = None
    response: str = None

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    latency_seconds: float = 0.0
    cost_usd: float = 0.0

    status: str = "SUCCESS"
    error_message: str = None

    start_time: datetime = None
    end_time: datetime = None

    timestamp: datetime = None