from pathlib import Path
import sqlite3

BASE_DIR = Path(__file__).resolve().parents[3]

DB_DIR = BASE_DIR / "storage"
DB_DIR.mkdir(exist_ok=True)

DB_PATH = DB_DIR / "llmops.db"

def export_execution(record):

    print("Storage Module: Writing execution record to storage")
    print("**************************************************")
    print(record)

    conn = sqlite3.connect(DB_PATH)

    conn.execute(
        """
        INSERT INTO agent_executions (
            trace_id,
            parent_trace_id,
            span_id,

            session_id,

            workflow_name,
            agent_name,
            model_name,

            prompt,
            response,

            prompt_tokens,
            completion_tokens,
            total_tokens,

            latency_seconds,
            cost_usd,

            status,
            error_message,

            start_time,
            end_time,

            timestamp
        )
        VALUES (
            ?, ?, ?,
            ?,
            ?, ?, ?,
            ?, ?,
            ?, ?, ?,
            ?, ?,
            ?, ?,
            ?, ?,
            ?
        )
        """,
        (
            getattr(record, "trace_id", None),
            getattr(record, "parent_trace_id", None),
            getattr(record, "span_id", None),

            getattr(record, "session_id", None),

            getattr(record, "workflow_name", None),
            getattr(record, "agent_name", None),
            getattr(record, "model_name", None),

            getattr(record, "prompt", None),
            getattr(record, "response", None),

            getattr(record, "prompt_tokens", 0),
            getattr(record, "completion_tokens", 0),
            getattr(record, "total_tokens", 0),

            getattr(record, "latency_seconds", 0.0),
            getattr(record, "cost_usd", 0.0),

            getattr(record, "status", "SUCCESS"),
            getattr(record, "error_message", None),

            str(getattr(record, "start_time", None)),
            str(getattr(record, "end_time", None)),

            str(getattr(record, "timestamp", None))
        )
    )

    conn.commit()
    conn.close()

    print("Record inserted successfully")