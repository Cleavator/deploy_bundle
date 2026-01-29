# CRC RAG + Droplet Analysis

A combined colorectal cancer (CRC) Q&A RAG assistant and droplet image analysis toolkit. The project includes:

- A Gradio web UI for chat + droplet inference.
- A literature-backed RAG pipeline with citations.
- An MCP (Model Context Protocol) server for agent platforms.

## Features

- **CRC RAG Q&A**: Retrieves evidence from a local vector index and generates cited answers.
- **Droplet analysis**: Predicts CRC/healthy status and concentration from microscopy images.
- **Gradio UI**: Simple local web app for clinicians or analysts.
- **MCP server**: Enables use as a tool in agent platforms (e.g., Cherry Studio).

## Repository Structure

- `crc_gradio_app.py` — Gradio UI entry point.
- `crc_qa_v2.py` — RAG logic.
- `mcp_server.py` — MCP server entry point.
- `drop/` — droplet analysis pipeline and models.
- `code/index.pkl` — vector index.
- `bib_db.json` — citation database.
- `outputs/` — example outputs.

## Prerequisites

- Python **3.10+**
- (Recommended) A virtual environment
- A valid **DEEPSEEK_API_KEY**

## Setup

### 1) Create and activate a virtual environment

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.lock.txt
```

> You can also use `requirements.txt` if you prefer flexible versions.

### 3) Download Model Files

The vector index files (`index.pkl`) are too large to host on GitHub.

1. Download the files from this [Google Drive link](https://drive.google.com/drive/folders/1Fvqv-Eu5pdn1Sp0ZcH94BK5KBxHTL99n?usp=drive_link).
2. Place the downloaded files (specifically `index.pkl`) into the `code/` directory.
   - Ensure the path is `code/index.pkl`.

### 4) Configure environment variables

Copy the example and add your key:

```bash
cp .env.example .env
```

Then edit `.env`:
```
DEEPSEEK_API_KEY=your_key_here
```

## Run the Gradio App

```bash
python crc_gradio_app.py
```

Open: http://localhost:7860

## Run the MCP Server

This exposes a single tool: `crc_rag_answer`.

```bash
python mcp_server.py
```

### Cherry Studio (MCP) Setup

1. Open **Cherry Studio** → **Settings** → **MCP Servers**.
2. Click **Add Server**.
3. Choose **Stdio**.
4. Command:
   - Windows: `python mcp_server.py`
   - macOS/Linux: `python3 mcp_server.py`
5. Working directory: the project root folder.
6. Environment variables:
   - `DEEPSEEK_API_KEY=your_key_here`

After connecting, use the tool name `crc_rag_answer`.

## Security Notes

Read the security guidance in `SECURITY_WARNING.md`. **Never commit `.env` or keys**.

## Troubleshooting

- If the RAG answers are empty, confirm `code/index.pkl` exists (see Setup step 3).
- If the API errors, verify `DEEPSEEK_API_KEY` is set.
- If GPU-related warnings appear, the app will still run on CPU.

## License

Add a license before publishing.
