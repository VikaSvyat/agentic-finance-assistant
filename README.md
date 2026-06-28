# Finance MCP Agent

An agentic finance dashboard that analyzes bank/card statements through MCP tools.

The app uses a Streamlit UI, an LLM-driven agent loop, and multiple MCP servers. The LLM decides which tool to call next, while deterministic Python/MCP tools do the actual data loading, calculations, report generation, filesystem work, and SQLite persistence.

## What It Does

- Analyze a single uploaded CSV statement.
- Analyze a directory of CSV statements as one combined financial view.
- Discover CSV files through the Filesystem MCP server.
- Merge multiple statements into `data/merged_statement.csv`.
- Normalize Hebrew bank/card statement columns into a common schema.
- Normalize spending categories, including splitting broad utility categories into fuel, electricity, gas, and utilities.
- Calculate total spending, transaction count, average transaction, top categories, and top merchants.
- Detect large or unusual expenses using merchant/category percentile thresholds.
- Generate savings advice from analysis outputs.
- Generate a clean monthly markdown report.
- Prepare and save monthly report data to SQLite through MCP.
- Show a technical agent trace in the Streamlit UI.

## Architecture

```text
Streamlit UI
    |
    v
app.py
    |
    v
run_agent(user_goal, csv_path, statements_dir)
    |
    v
Agent loop
    |
    +--> LLM chooses exactly one next action as JSON
    |
    +--> MCPManager calls the selected MCP tool
    |
    +--> Full tool output is stored in Python memory
    |
    +--> Short observation is sent back to the LLM
    |
    v
final_answer returns the generated monthly report
```

The MCP manager connects three tool providers:

```text
MCPManager
    |
    +-- filesystem: @modelcontextprotocol/server-filesystem
    |
    +-- sqlite: mcp-sqlite
    |
    +-- finance: custom Python MCP server
```

## Main Components

### `app.py`

Streamlit dashboard and user entry point.

Responsibilities:

- lets the user choose `Single CSV` or `Directory` mode;
- saves uploaded CSV files into `DATA_DIR`;
- calls `run_agent(...)`;
- renders KPI cards, category tables, merchant tables, unusual expenses, and savings insights;
- displays the raw agent trace for debugging and demo purposes.

### `agent/agent.py`

The agent orchestrator.

Responsibilities:

- builds the system prompt and user task;
- asks the LLM for one next action at a time;
- parses the LLM's JSON action;
- validates tool ordering and required previous outputs;
- enriches or fixes tool arguments before execution;
- stores important full outputs in internal memory;
- sends compact observations back to the LLM to reduce token usage;
- returns the final monthly report once the workflow is complete.

### `agent/mcp_manager.py`

MCP connection manager.

Responsibilities:

- starts MCP servers over stdio;
- lists available tools;
- stores tool metadata under names like `finance.analyze_statement`;
- dispatches tool calls to the right MCP session;
- closes all MCP sessions cleanly.

### `agent/llm.py`

Small LLM provider wrapper.

Supported providers:

- Groq
- Ollama

### `servers/finance_server.py`

Custom Finance MCP server.

Available finance tools include:

- `categorize_transactions`
- `analyze_statement`
- `get_category_breakdown`
- `get_top_merchants`
- `find_unusual_expenses`
- `generate_savings_advice`
- `generate_monthly_report`
- `prepare_monthly_report_record`
- `merge_statements`

## Data Flow

### Single CSV Mode

```text
Upload CSV
    |
    v
Save file into DATA_DIR
    |
    v
finance.analyze_statement
    |
    v
finance.find_unusual_expenses
    |
    v
finance.generate_savings_advice
    |
    v
finance.generate_monthly_report
    |
    v
finance.prepare_monthly_report_record
    |
    v
sqlite.create_record
    |
    v
Streamlit dashboard output
```

### Directory Mode

```text
Select statements directory
    |
    v
filesystem.list_directory
    |
    v
finance.merge_statements
    |
    v
Analyze merged CSV using the same single-file finance tools
    |
    v
Generate report and save SQLite record
```

## Project Structure

```text
finance-mcp-agent/
├── agent/
│   ├── agent.py
│   ├── llm.py
│   └── mcp_manager.py
├── client/
│   ├── list_tools.py
│   └── test_finance_client.py
├── data/
│   ├── statements/
│   ├── merged_statement.csv
│   └── sample_statement.csv
├── database/
├── logs/
├── servers/
│   └── finance_server.py
├── app.py
├── requirements.txt
└── README.md
```

## Environment Variables

Create a `.env` file in the project root.

```bash
LLM_PROVIDER=groq
MODEL=your-groq-model
GROQ_API_KEY=your-api-key

DATA_DIR=./data
DB_PATH=./database/finance.db
LOG_FILE=./logs/agent.log

FAST_DEV_MODE=false
FAST_DEV_SKIP_SQLITE=false
MAX_AGENT_STEPS=14
MAX_TOOL_OUTPUT_LENGTH=1200
MAX_HISTORY_MESSAGES=6
MAX_TRACE_OUTPUT_LENGTH=3000

FINANCE_CACHE_ENABLED=true
FINANCE_CACHE_MAX_ENTRIES=16
```

For Ollama:

```bash
LLM_PROVIDER=ollama
MODEL=your-ollama-model
OLLAMA_URL=http://localhost:11434
```

## Running the Project

Install Python dependencies:

```bash
pip install -r requirements.txt
```

The app also starts Node-based MCP servers through `npx`, so Node.js/npm must be available.

Run the Streamlit app:

```bash
streamlit run app.py
```

## Streamlit Community Cloud Deployment

The app can run on Streamlit Community Cloud while keeping the current MCP architecture.
The deployment uses Python MCP code plus Node-based MCP servers started through `npx`.

### Cloud files

The repository includes:

- `requirements.txt` for Python dependencies.
- `packages.txt` for system packages required by the Node-based MCP servers.

`packages.txt` must contain:

```text
nodejs
npm
```

### Streamlit Secrets

Configure these values in the Streamlit Cloud app settings.
Do not commit real secrets to the repository.

```toml
LLM_PROVIDER = "groq"
MODEL = "llama-3.1-8b-instant"
GROQ_API_KEY = "your-groq-api-key"

DATA_DIR = "./data"
DB_PATH = "./database/finance.db"

FAST_DEV_MODE = false
FAST_DEV_SKIP_SQLITE = false
```

The application copies supported Streamlit secrets into environment variables when
the corresponding environment variable is not already set. Local `.env` files still
work for development.

### Deployment limitations

- Groq is required in Streamlit Cloud. Ollama is local-only because the cloud app
  cannot access a local `localhost:11434` Ollama server.
- SQLite is used as demo/session storage. It is suitable for project demos, but it
  should not be treated as durable production storage on Streamlit Cloud.
- Node/npm are required because the app starts the filesystem and SQLite MCP
  servers through `npx`.

## Developer Utilities

List SQLite MCP tools:

```bash
python client/list_tools.py
```

Run the direct finance MCP client demo:

```bash
python client/test_finance_client.py
```

## Runtime Tuning

The agent runtime is configurable through environment variables.

`FAST_DEV_MODE=true` switches the defaults to a shorter development loop:

- fewer maximum agent steps;
- shorter tool output previews sent back to the LLM;
- fewer runtime history messages kept in the prompt.

`FAST_DEV_SKIP_SQLITE=true` skips SQLite persistence during development and returns the final report immediately after `finance.generate_monthly_report`.

The architecture stays the same: the LLM still selects MCP tools one step at a time, and MCP tools still perform the work.

Configurable runtime limits:

- `MAX_AGENT_STEPS`: maximum LLM/tool iterations before the agent stops.
- `MAX_TOOL_OUTPUT_LENGTH`: maximum characters from a tool result included in the LLM-facing observation preview.
- `MAX_HISTORY_MESSAGES`: number of recent runtime messages kept after the initial system/user context.
- `MAX_TRACE_OUTPUT_LENGTH`: maximum characters shown for large values in the Streamlit technical trace.
- `FAST_DEV_SKIP_SQLITE`: skip `finance.prepare_monthly_report_record` and `sqlite.create_record` in dev runs.

Timing instrumentation is recorded for each step:

- LLM response time;
- MCP tool execution time;
- total step duration.

These timings are written to `LOG_FILE` and shown in the Streamlit technical agent trace.

## Performance Bottlenecks Found

Main latency sources in the current runtime:

- LLM calls: every agent step requires a model round trip, so unnecessary iterations are expensive.
- MCP startup and tool calls: each run connects filesystem, SQLite, and finance MCP servers, then dispatches tool calls over stdio.
- Repeated file processing: finance tools reload and reparse CSV files independently for analysis, unusual expense detection, categorization, and merge workflows.
- Prompt growth: observations and action history can grow across iterations if not pruned.
- Large tool outputs: full JSON outputs can be large, especially unusual-expense payloads or directory listings.
- Logging and trace payloads: full observations are useful for debugging but can become noisy during fast iteration.

Current mitigations:

- configurable maximum agent steps;
- configurable tool output preview length;
- configurable prompt history limit;
- compact LLM observations while keeping full tool outputs in Python memory;
- per-step timing metrics to identify whether a slow run is caused by the LLM or by a tool.

## Safe Caching Opportunities

The Finance MCP server includes a small in-process cache for parsed and normalized CSV statements.

Implemented cache:

- `load_bank_csv(csv_path)` caches parsed CSV DataFrames;
- `load_normalized_bank_csv(csv_path)` caches parsed DataFrames with normalized categories;
- cache keys include absolute path, file size, and file modified time;
- cached DataFrames are returned as copies so tool code cannot mutate shared cached state;
- `FINANCE_CACHE_ENABLED=false` disables this cache;
- `FINANCE_CACHE_MAX_ENTRIES` controls the maximum number of cached statement versions.

Safe future opportunities:

- cache `finance.analyze_statement(csv_path)` for unchanged files;
- cache `finance.get_category_breakdown(csv_path)` and `finance.get_top_merchants(csv_path)` for unchanged files;
- cache `finance.merge_statements(csv_files)` by input file paths and modified times;
- reuse LLM provider clients across calls to avoid recreating clients every step.

Caching should avoid stale financial results. Any cache key should include file path, file size, and modified time at minimum.

## Current Status

Implemented:

- Streamlit finance dashboard
- Single CSV analysis
- Directory-based multi-statement analysis
- Filesystem MCP discovery
- Multi-file statement merge
- Custom Finance MCP server
- LLM tool orchestration
- Transaction/category normalization
- Category and merchant aggregation
- Unusual expense detection
- Savings advice generation
- Monthly report generation
- SQLite record preparation and persistence
- Agent trace/debug view

Possible next improvements:

- month-over-month comparisons;
- recurring subscription detection;
- budget targets by category;
- historical trend charts from SQLite;
- report export/download controls;
- stronger automated tests around the agent loop and finance tools.
