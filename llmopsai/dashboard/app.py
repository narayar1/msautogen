import streamlit as st

st.set_page_config(
    page_title="LLMOps Dashboard",
    layout="wide"
)

st.title("LLMOps Observability Dashboard")

st.markdown("""
Use the navigation menu to explore:

- Agent Executions
- Token Usage
- Latency Analysis
""")