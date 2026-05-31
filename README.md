# Finance MCP Agent

An agentic AI application that analyzes bank statements using the Model Context Protocol (MCP).

The project demonstrates how an LLM can orchestrate multiple MCP servers to perform financial analysis, generate insights, create reports, and persist results in SQLite.

## Features

### Finance MCP Server

Custom MCP server providing financial analysis tools:

* Analyze monthly bank statements
* Categorize transactions
* Detect unusual expenses
* Generate savings advice
* Generate monthly financial reports

### Filesystem MCP

Uses the official MCP Filesystem Server to:

* Read files
* Write reports
* Manage report storage

### SQLite MCP

Uses the official MCP SQLite Server to:

* Store monthly reports
* Store category summaries
* Persist historical financial data

### Agent Orchestration

The agent automatically:

1. Analyzes a statement
2. Detects unusual expenses
3. Generates savings advice
4. Creates a monthly report
5. Saves report metadata to SQLite

The LLM decides which tool to call next and executes one action at a time.

## Architecture

```text
Streamlit UI
      |
      v
Agent Loop
      |
      +--------------------+
      |                    |
      v                    v
Finance MCP          Official MCP Servers
(Server)             (Filesystem + SQLite)
      |
      v
Financial Analysis
```

## Project Structure

```text
finance-mcp-agent/

├── agent/
│   ├── agent.py
│   ├── llm.py
│   └── mcp_manager.py
│
├── client/
│
├── servers/
│   └── finance_server.py
│
├── data/
│   └── sample_statement.csv
│
├── database/
│
├── logs/
│
├── app.py
├── requirements.txt
└── README.md
```

## Technologies

* Python
* MCP (Model Context Protocol)
* Streamlit
* SQLite
* Pandas
* Groq LLM API

## Current Functionality

Implemented:

* Custom Finance MCP server
* MCP tool orchestration
* Transaction categorization
* Unusual expense detection
* Savings advice generation
* Monthly report generation
* SQLite persistence
* Category-level aggregation

Planned:

* Multi-file statement processing
* Month-over-month analysis
* Recurring subscription detection
* Budget recommendations
* Historical trend analysis
* RAG integration

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure environment variables:

```bash
cp .env.example .env
```

Run:

```bash
streamlit run app.py
```

## Example Workflow

1. Upload a bank statement CSV
2. Agent analyzes transactions
3. Agent identifies unusual expenses
4. Agent generates savings advice
5. Agent creates a monthly report
6. Results are stored in SQLite


