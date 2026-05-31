import os
import asyncio
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agent.agent import run_agent

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Finance MCP Agent", layout="wide")

st.title("Finance MCP Agent")
st.write("Upload a monthly bank statement CSV and let the LLM orchestrate MCP tools.")

uploaded_file = st.file_uploader(
    "Upload CSV bank statement",
    type=["csv"],
)

default_goal = """
Analyze this monthly bank statement.
Find spending by category, unusual expenses, savings advice,
save the report, and store summary in SQLite.
"""

goal = st.text_area("User goal", value=default_goal, height=140)

if uploaded_file:
    csv_path = DATA_DIR / uploaded_file.name

    with open(csv_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"File saved to: {csv_path}")

    if st.button("Run Agent"):
        with st.spinner("Agent is working..."):
            result = asyncio.run(run_agent(goal, str(csv_path)))

        st.subheader("Final Answer")
        st.write(result["answer"])

        st.subheader("Agent Steps")

        for step in result["steps"]:
            with st.expander(f"Step {step['step']}: {step['tool']}"):
                st.write("Reason:")
                st.write(step.get("reason", ""))

                if "args" in step:
                    st.write("Args:")
                    st.json(step["args"])

                if "observation" in step:
                    st.write("Observation:")
                    st.json(step["observation"])
else:
    st.info("Upload a CSV file first.")