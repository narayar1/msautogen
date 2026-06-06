import sqlite3

conn = sqlite3.connect("storage/llmops.db")

conn.execute("""drop table if exists agent_executions""")

conn.execute("""

CREATE TABLE IF NOT EXISTS agent_executions (
     trace_id TEXT,
     parent_trace_id TEXT,
    span_id TEXT,

    session_id TEXT,

    workflow_name TEXT,
    agent_name TEXT,
    model_name TEXT,

    prompt TEXT,
    response TEXT,

    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,

    latency_seconds REAL,
    cost_usd REAL,

    status TEXT,
    error_message TEXT,

    start_time TEXT,
    end_time TEXT,

    timestamp TEXT
);
""")

conn.commit()
conn.close()

print("Table created")