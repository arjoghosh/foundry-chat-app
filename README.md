# Foundry Chat

A ChatGPT-style developer chat application built with **Python**, **Streamlit**, and an **Azure OpenAI deployment in Microsoft Foundry**.

Chat with your model, upload a ZIP of source files for analysis, save conversations locally with SQLite, and monitor response duration and token usage.

> Designed for trusted, local, single-user use. Authentication and user-isolated storage are not included.

## Features

- Streaming responses with Markdown formatting
- Syntax-highlighted code blocks with native copy controls
- Dark theme
- Multiple persistent conversations stored in SQLite
- Retry and regenerate the latest response
- Markdown conversation export
- Response duration, time to first text, and API-reported token usage
- Daily request limits and token-budget reservations
- Per-request input and output limits
- Request cooldowns
- ZIP source-code uploads with file selection and context preview
- Persistent attachment snapshots saved with admitted questions
- ZIP validation without extracting files to the filesystem or executing code

## Requirements

- Python 3.10 or newer
- An Azure OpenAI deployment that supports **Chat Completions**
- Its resource endpoint, API key, and exact deployment name
- Internet access to the configured endpoint

SQLite is included with standard Python installations through the `sqlite3` module. No separate database server is required.

## Project Structure

```text
foundry-chat/
├── app.py
├── attachments.py
├── storage.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml
└── data/
    └── chat.db
```

The `data/` directory and database are created automatically.

## Quick Start

### 1. Clone the repository

Replace the URL with your repository:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The current dependencies are:

```text
streamlit>=1.40.0,<2.0.0
openai>=1.55.0,<3.0.0
```

ZIP processing, hashing, and SQLite use Python's standard library.

### 4. Configure your deployment

Create `.streamlit/secrets.toml`:

```toml
AZURE_OPENAI_ENDPOINT = "https://YOUR-RESOURCE.openai.azure.com/openai/v1/"
AZURE_OPENAI_API_KEY = "YOUR-API-KEY"
AZURE_OPENAI_DEPLOYMENT = "YOUR-DEPLOYMENT-NAME"

AZURE_OPENAI_INCLUDE_USAGE = true
AZURE_OPENAI_OUTPUT_PARAMETER = "max_completion_tokens"

[usage_limits]
daily_requests = 100
daily_token_budget = 100000
max_input_tokens = 16000
max_output_tokens = 4000
min_seconds_between_requests = 2
```

Important:

- Use the **Azure OpenAI v1 endpoint**, not a Foundry project endpoint.
- Use the **exact deployment name**, which may differ from the model name.
- This app uses `OpenAI(base_url=...)`; an API-version setting is not required for this configuration.
- The deployment must support Chat Completions.
- Set `AZURE_OPENAI_INCLUDE_USAGE = false` if the deployment rejects streaming usage options.
- Use `"max_tokens"` instead of `"max_completion_tokens"` only if required by your deployment.

Environment variables take precedence over Streamlit secrets.

Usage-limit environment variables use the `USAGE_` prefix:

```text
USAGE_DAILY_REQUESTS
USAGE_DAILY_TOKEN_BUDGET
USAGE_MAX_INPUT_TOKENS
USAGE_MAX_OUTPUT_TOKENS
USAGE_MIN_SECONDS_BETWEEN_REQUESTS
```

### 5. Configure the dark theme

Create `.streamlit/config.toml`:

```toml
[theme]
base = "dark"
primaryColor = "#10a37f"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#171c26"
textColor = "#ececf1"
font = "sans serif"

[server]
address = "127.0.0.1"
```

Binding to `127.0.0.1` keeps the app accessible from the local machine rather than exposing it on all network interfaces.

### 6. Start the app

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Using the App

### Chat

1. Start a new conversation.
2. Enter your question.
3. View the streamed response.
4. Hover over a code block to use its copy control.

Conversation messages are saved locally in SQLite.

### Analyze uploaded source files

1. Open **Attach source files from a ZIP**.
2. Upload a project ZIP.
3. Select the relevant files.
4. Optionally preview the attachment context.
5. Ask a question or use a suggested review prompt.

Example questions:

- “Explain how these files work together.”
- “Review the database transaction handling.”
- “Find potential bugs and cite relevant filenames and line numbers.”
- “Generate tests for this module.”

The model does not receive or unpack the ZIP itself. Python reads selected supported text files, and the app includes their contents in the model request.

Uploaded code is not executed.

### Attachment behavior

- No files are selected automatically.
- Draft uploads are session-based.
- Selected file snapshots become persistent when a request is admitted and its question is saved.
- Retry uses the original question's saved attachments.
- Earlier attachments are included only when their messages are within the selected conversation-history window.
- Removing a draft upload does not remove attachments already saved with messages.
- Reattaching the same files can duplicate context and increase usage.
- Markdown export includes attachment filenames, not attachment contents.

Retry preserves attachment snapshots but uses the current system instructions and history setting.

## Supported Uploads

ZIP archives may contain supported UTF-8 source and text files, including:

- Python, JavaScript, TypeScript, Java, C/C++, C#, Go, Rust, and other supported source files
- Markdown and plain text
- SQL and shell scripts
- JSON, YAML, TOML, XML, and CSV
- Common project files such as `Dockerfile`, `Makefile`, and `requirements.txt`

PDFs, images, compiled binaries, and nested archives are not processed as model context.

The exact allowlist is defined in `attachments.py`.

### Default ZIP limits

| Limit | Default |
|---|---:|
| Compressed ZIP size | 10 MB |
| Archive entries | 2,000 |
| Accepted text files | 200 |
| Individual text-file size | 256 KB |
| Total bytes read from candidate text entries | 2 MB |
| Maximum accepted compression ratio | 200:1 |

Additional validation includes:

- Rejecting unsafe archive paths
- Rejecting symbolic links and special entries
- Rejecting encrypted entries
- Rejecting duplicate or case-conflicting paths
- Skipping common dependency and build directories
- Skipping common credential filenames
- Enforcing bounded reads during decompression

These checks reduce risk but do not constitute antivirus scanning or guarantee that uploaded content is safe.

## Response Metrics

Successful responses display available metrics:

```text
⏱ 4.28 s · Input: 1,240 · Output: 386 · Total: 1,626
```

- **Duration:** elapsed time for the model call and stream.
- **Time to first text:** elapsed time until the first nonempty text chunk.
- **Token usage:** counts reported by the API, when available.

Input usage includes instructions, included conversation history, attachment context, and the current question.

Missing token usage is displayed as unavailable, not zero.

## Usage Limits

Limits are checked before starting each model request.

| Control | Behavior |
|---|---|
| Daily request limit | Counts admitted attempts, including retries and failures |
| Daily token budget | Reserves estimated input plus maximum output before sending |
| Input limit | Checks the app's conservative input estimate |
| Output limit | Sends the configured output-token limit to the deployment |
| Cooldown | Enforces a minimum interval between admitted requests |

Daily accounting uses **UTC**.

### Token accounting

1. Build the full request, including attachments.
2. Estimate input size.
3. Atomically check limits and reserve budget in SQLite.
4. Send the request.
5. Replace the reservation with API-reported total usage when available.
6. Retain the reservation if final usage is unavailable.

Deleting or clearing a conversation does **not** refund its usage.

> Token-budget enforcement is an application-level safeguard, not a guaranteed Azure billing cap. Estimates can differ from provider accounting, failed requests may incur usage, and other applications can use the same deployment.

The input estimate uses UTF-8 byte length plus overhead. It is deliberately conservative and is not an exact model tokenizer count. Large files may reach the configured input limit sooner than expected.

## Local Persistence

SQLite stores:

- Conversation titles and messages
- Selected attachment snapshots
- Response metrics
- Generation attempts
- Usage reservations and accounting

The database is located at:

```text
data/chat.db
```

SQLite may also create WAL and shared-memory files in the same directory.

Important:

- All browser sessions share the same database.
- SQLite is not encrypted by default.
- Use durable local storage.
- Use SQLite's backup API for consistent backups while the app is running.
- Deleting records is not a secure-erasure guarantee; copies may remain in backups or database storage until handled separately.

## Recovering Interrupted Requests

If the server stops during generation, an attempt may remain marked as running.

Stop Streamlit first, then run:

```bash
python storage.py --recover
```

Restart:

```bash
streamlit run app.py
```

Recovery marks unfinished attempts as interrupted. It does not resend requests or remove their unconfirmed token reservations.

Do not run recovery while another app process is actively generating responses.

## Troubleshooting

### HTTP 404

Check:

- The exact deployment name
- The Azure OpenAI resource endpoint
- The `/openai/v1/` endpoint suffix
- Chat Completions support

Do not use the Foundry project endpoint.

### HTTP 400

Check:

- Whether the deployment supports the configured output-token parameter
- Whether it supports `stream_options={"include_usage": True}`
- Request/context size
- Content restrictions

### Token usage unavailable

The deployment or interrupted stream may not return final usage metadata. Timing still works, and the usage ledger retains an unconfirmed reservation.

### Input limit reached after selecting files

Select fewer or smaller files, reduce conversation history, or adjust the configured input limit within your deployment's supported context capacity.

### ZIP file skipped or rejected

Check `attachments.py` limits and the skipped-entry list. Only supported UTF-8 text files are processed.

### SQLite import fails

Verify your Python installation:

```bash
python -c "import sqlite3; print(sqlite3.sqlite_version)"
```

Some custom Python builds omit SQLite support. Fix the Python installation rather than adding `sqlite3` to `requirements.txt`.

## Security and Privacy

Never commit:

- API keys or credentials
- `.streamlit/secrets.toml`
- `.env` files
- `data/`
- Database backups containing conversations or source code

Recommended `.gitignore` entries:

```gitignore
.streamlit/*
!.streamlit/config.toml

data/
backups/

.env
.env.*
!.env.example

.venv/
venv/
__pycache__/
*.py[cod]
.DS_Store
```

Additional precautions:

- Review selected source files before sending them to the model service.
- Credential-file filtering is not comprehensive secret detection.
- PII redaction and antivirus scanning are not included.
- Uploaded content is treated as untrusted reference material, but prompt-injection defenses are not a guarantee.
- Do not execute uploaded or model-generated code on the app host.
- Restrict access to the local machine and use disk encryption when needed.
- Add authentication, authorization, and user-isolated storage before exposing the app publicly.
- Rotate any credential accidentally committed to Git, even if it is later removed.

## Current Limitations

- Local, single-user design
- No authentication or per-user storage isolation
- Chat Completions integration only
- No code execution or test runner
- No automatic repository retrieval or semantic search
- No native PDF or image analysis
- Successful regeneration replaces the displayed answer; earlier attempt metrics remain in the ledger
- Draft uploads do not survive a new browser session
- No total database-size quota or automatic retention policy

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

Third-party dependencies remain subject to their respective licenses.
Microsoft Foundry and Azure OpenAI services are subject to Microsoft's
applicable terms and usage charges.

This is an independent project and is not affiliated with or endorsed by
Microsoft or OpenAI.