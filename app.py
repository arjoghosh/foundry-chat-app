import hashlib
import logging
import os
import time

from pathlib import Path

import streamlit as st

from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

import storage
from attachments import (
    ATTACHMENT_INSTRUCTIONS,
    AttachmentError,
    MAX_ZIP_BYTES,
    build_attachment_context,
    read_zip_text_files,
    to_api_message,
)


# -------------------------
# Page configuration
# -------------------------
APP_DIR = Path(__file__).resolve().parent
ICON_PATH = APP_DIR / "assets" / "velora-icon.png"

st.logo(
    str(ICON_PATH),
    icon_image=str(ICON_PATH),
)

st.set_page_config(
    page_title="Velora",
    page_icon=str(ICON_PATH),
    layout="centered",
)

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """
You are a helpful, accurate AI assistant.

Use Markdown for readability.
Put code in fenced code blocks with the appropriate language tag.
Keep explanations outside code blocks.
Provide runnable examples when appropriate.
Be clear about uncertainty and do not invent facts.
""".strip()

st.markdown(
    """
    <style>
    .block-container {
        max-width: 960px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(217, 154, 115, 0.14);
    }

    [data-testid="stChatMessage"] {
        background-color: #211C28;
        border: 1px solid rgba(217, 154, 115, 0.12);
        border-radius: 18px;
        margin-bottom: 1rem;
        padding: 1.25rem;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        border-radius: 12px;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        border-color: #D99A73;
        box-shadow: 0 0 0 1px rgba(217, 154, 115, 0.18);
    }

    [data-testid="stChatInput"] {
        border-radius: 16px;
        border: 1px solid rgba(217, 154, 115, 0.25);
    }

    h1, h2, h3 {
        letter-spacing: -0.025em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Configuration
# -------------------------
def get_setting(name, default=""):
    value = os.getenv(name)

    if value is not None:
        return value

    try:
        return st.secrets.get(name, default)
    except FileNotFoundError:
        return default


def load_limits():
    defaults = {
        "daily_requests": 100,
        "daily_token_budget": 100000,
        "max_input_tokens": 16000,
        "max_output_tokens": 4000,
        "min_seconds_between_requests": 2,
    }

    try:
        configured = st.secrets.get("usage_limits", {})
    except FileNotFoundError:
        configured = {}

    limits = {}

    for name, default in defaults.items():
        value = int(
            os.getenv(
                f"USAGE_{name.upper()}",
                configured.get(name, default),
            )
        )

        minimum = 0 if name == "min_seconds_between_requests" else 1

        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}.")

        limits[name] = value

    return limits


endpoint = str(get_setting("AZURE_OPENAI_ENDPOINT"))
api_key = str(get_setting("AZURE_OPENAI_API_KEY"))
deployment = str(get_setting("AZURE_OPENAI_DEPLOYMENT"))

include_usage = (
    str(get_setting("AZURE_OPENAI_INCLUDE_USAGE", True)).strip().lower()
    in {"true", "1", "yes", "on"}
)

output_parameter = str(
    get_setting(
        "AZURE_OPENAI_OUTPUT_PARAMETER",
        "max_completion_tokens",
    )
)

if not all([endpoint, api_key, deployment]):
    st.error(
        "Configure AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, "
        "and AZURE_OPENAI_DEPLOYMENT."
    )
    st.stop()

if output_parameter not in {"max_completion_tokens", "max_tokens"}:
    st.error(
        "AZURE_OPENAI_OUTPUT_PARAMETER must be "
        "'max_completion_tokens' or 'max_tokens'."
    )
    st.stop()

try:
    limits = load_limits()
except (ValueError, TypeError) as exc:
    st.error(f"Invalid usage-limit configuration: {exc}")
    st.stop()


@st.cache_resource
def create_client(endpoint, api_key):
    return OpenAI(
        base_url=endpoint.rstrip("/") + "/",
        api_key=api_key,
        timeout=90.0,
        # Keep retries visible and separately accounted for.
        max_retries=0,
    )


@st.cache_resource
def initialize_database():
    storage.initialize()
    return True


initialize_database()
client = create_client(endpoint, api_key)


# -------------------------
# Helpers
# -------------------------
def estimate_input_tokens(messages):
    """
    Conservative text-only budgeting heuristic.

    UTF-8 byte length is deliberately more cautious than chars / 4.
    Added overhead allows for message framing. This is not an exact
    tokenizer count or a guaranteed bound for every deployment.
    """
    return 128 + sum(
        len(message["content"].encode("utf-8")) + 32
        for message in messages
    )


def metrics_summary(metrics):
    if not metrics:
        return ""

    parts = []

    duration = metrics.get("duration_seconds")
    if duration is not None:
        parts.append(f"⏱ {duration:.2f} s")

    if metrics.get("usage_available"):
        for label, key in [
            ("Input", "input_tokens"),
            ("Output", "output_tokens"),
            ("Total", "total_tokens"),
        ]:
            value = metrics.get(key)

            if value is not None:
                parts.append(f"{label}: {value:,}")
    else:
        parts.append("Token usage unavailable")

    return " · ".join(parts)


def export_markdown(chat):
    sections = [f"# {chat['title']}", ""]

    for message in chat["messages"]:
        role = "You" if message["role"] == "user" else "Assistant"
        sections.append(f"## {role}\n\n{message['content']}\n")

        attached = message.get("attachments", [])

        if attached:
            sections.append("Attached files:\n")

            for item in attached:
                sections.append(f"- {item['path']}")

            sections.append(
                "\n*Attachment contents are stored in the local database "
                "and are not included in this export.*\n"
            )

        summary = metrics_summary(message.get("metrics", {}))

        if summary:
            sections.append(f"*{summary}*\n")

    return "\n".join(sections)


def friendly_error(exc):
    if isinstance(exc, AuthenticationError):
        return "Authentication failed. Check the API key and endpoint."

    if isinstance(exc, RateLimitError):
        return "The deployment is rate-limited. Wait and retry."

    if isinstance(exc, APIConnectionError):
        return (
            "Could not reach the deployment. "
            "Check your network and endpoint."
        )

    if isinstance(exc, APIStatusError):
        if exc.status_code == 404:
            return (
                "Deployment or route not found. Check the deployment name "
                "and /openai/v1/ endpoint."
            )

        if exc.status_code == 400:
            return (
                "The deployment rejected the request. Check context size, "
                "Chat Completions support, streaming usage support, and "
                "the configured output-token parameter."
            )

        return f"The deployment returned HTTP {exc.status_code}."

    if isinstance(exc, RuntimeError):
        return str(exc)

    return "An unexpected error occurred. Please retry."


def stream_answer(messages, metrics):
    started = time.perf_counter()
    stream = None

    metrics.update(
        {
            "deployment": deployment,
            "duration_seconds": None,
            "time_to_first_text_seconds": None,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "usage_available": False,
        }
    )

    options = {
        output_parameter: limits["max_output_tokens"],
    }

    if include_usage:
        options["stream_options"] = {"include_usage": True}

    try:
        stream = client.chat.completions.create(
            model=deployment,
            messages=messages,
            stream=True,
            **options,
        )

        for chunk in stream:
            # Usage can arrive in a final chunk with no choices.
            usage = getattr(chunk, "usage", None)

            if usage is not None:
                metrics.update(
                    {
                        "input_tokens": usage.prompt_tokens,
                        "output_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens,
                        "usage_available": usage.total_tokens is not None,
                    }
                )

            if not chunk.choices:
                continue

            choice = chunk.choices[0]

            if choice.finish_reason == "content_filter":
                raise RuntimeError(
                    "The response was stopped by the deployment's safety filter."
                )

            if choice.delta.content:
                if metrics["time_to_first_text_seconds"] is None:
                    metrics["time_to_first_text_seconds"] = (
                        time.perf_counter() - started
                    )

                yield choice.delta.content

            if choice.finish_reason == "length":
                yield (
                    "\n\n---\n"
                    "*Output limit reached. Ask me to continue if needed.*"
                )

    finally:
        metrics["duration_seconds"] = time.perf_counter() - started

        if stream is not None:
            stream.close()


def show_notice(message):
    st.session_state.notice = message


def queue_request(chat, prompt=None):
    """
    Snapshot selected files for a new question.

    Retry does not use the current upload selection; it uses the
    attachment snapshots already saved with the original question.
    """
    selected = []

    if prompt is not None:
        selected = [
            dict(item)
            for item in st.session_state.get(
                f"selected_attachments_{chat['id']}",
                [],
            )
        ]

    st.session_state.pending_job = {
        "chat_id": chat["id"],
        "revision": chat["revision"],
        "prompt": prompt,
        "attachments": selected,
    }


def reset_attachment_draft(chat_id):
    """
    Clear draft attachment state.

    Incrementing the widget version gives the uploader a fresh key.
    Saved message attachments are managed separately in SQLite.
    """
    version_key = f"attachment_version_{chat_id}"

    st.session_state[version_key] = (
        st.session_state.get(version_key, 0) + 1
    )
    st.session_state.pop(f"selected_attachments_{chat_id}", None)
    st.session_state.pop(f"parsed_zip_{chat_id}", None)


def render_saved_attachments(message):
    saved_attachments = message.get("attachments", [])

    if not saved_attachments:
        return

    with st.expander(f"📎 Attached files ({len(saved_attachments)})"):
        for item in saved_attachments:
            st.text(
                f"{item['path']} — "
                f"{item.get('size_bytes', 0):,} bytes"
            )

        st.caption(
            "These file snapshots are stored with this question "
            "and reused when retrying."
        )


def render_zip_attachments(chat, busy):
    """
    Render the ZIP picker and update the draft selection.

    Parsed ZIP contents are cached only in this browser session, not in
    a global Streamlit cache. Admitted questions persist their selected
    file snapshots in SQLite.
    """
    chat_id = chat["id"]
    selection_state_key = f"selected_attachments_{chat_id}"
    cache_key = f"parsed_zip_{chat_id}"
    version = st.session_state.get(f"attachment_version_{chat_id}", 0)

    selected_files = []

    with st.expander("📎 Attach source files from a ZIP", expanded=False):
        st.warning(
            "Selected file contents will be sent to your configured model "
            "service when you submit a question. Review them for secrets first. "
            "Automatic filtering cannot detect every credential."
        )

        uploaded_zip = st.file_uploader(
            "Upload a source-code ZIP",
            type=["zip"],
            accept_multiple_files=False,
            key=f"zip_upload_{chat_id}_{version}",
            disabled=busy,
            help=(
                "Up to 10 MB compressed. Supported UTF-8 text files only. "
                "256 KB per file and 2 MB total supported content."
            ),
        )

        if uploaded_zip is not None:
            if uploaded_zip.size > MAX_ZIP_BYTES:
                st.error("ZIP exceeds the 10 MB upload limit.")
                st.session_state.pop(cache_key, None)

            else:
                try:
                    zip_bytes = uploaded_zip.getvalue()
                    zip_hash = hashlib.sha256(zip_bytes).hexdigest()

                    cached = st.session_state.get(cache_key)

                    if not cached or cached["sha256"] != zip_hash:
                        # Remove stale parsed content before inspecting a new ZIP.
                        st.session_state.pop(cache_key, None)

                        files, skipped = read_zip_text_files(zip_bytes)

                        cached = {
                            "sha256": zip_hash,
                            "files": files,
                            "skipped": skipped,
                        }
                        st.session_state[cache_key] = cached

                    files = cached["files"]
                    skipped = cached["skipped"]

                    if not files:
                        st.info("No supported UTF-8 text files were found.")

                    else:
                        by_path = {
                            item["path"]: item
                            for item in files
                        }

                        selected_paths = st.multiselect(
                            "Files to include with your next question",
                            options=list(by_path),
                            default=[],
                            key=(
                                f"zip_selection_{chat_id}_"
                                f"{version}_{zip_hash}"
                            ),
                            disabled=busy,
                            help=(
                                "Start with a few relevant files. Selecting "
                                "the entire project may exceed your input limit."
                            ),
                        )

                        selected_files = [
                            dict(by_path[path])
                            for path in selected_paths
                        ]

                        total_bytes = sum(
                            item["size_bytes"]
                            for item in selected_files
                        )

                        st.caption(
                            f"{len(files)} supported files found · "
                            f"{len(selected_files)} selected · "
                            f"{total_bytes:,} selected bytes"
                        )

                        if selected_files:
                            context = build_attachment_context(selected_files)

                            # This excludes the final question, history,
                            # and system instructions.
                            attachment_estimate = estimate_input_tokens(
                                [{"role": "user", "content": context}]
                            )

                            st.caption(
                                "Attachment-only input estimate: "
                                f"{attachment_estimate:,}. "
                                "The final limit check also includes your "
                                "question, instructions, and conversation history."
                            )

                            if (
                                attachment_estimate
                                > limits["max_input_tokens"]
                            ):
                                st.warning(
                                    "These attachments already exceed the "
                                    "configured input estimate limit. "
                                    "Select fewer or smaller files."
                                )

                            if st.checkbox(
                                "Preview the text that will be included",
                                key=(
                                    f"zip_preview_{chat_id}_"
                                    f"{version}_{zip_hash}"
                                ),
                                disabled=busy,
                            ):
                                st.code(context, language="text")

                    if skipped:
                        st.caption(f"{len(skipped)} entries skipped.")

                        if st.checkbox(
                            "Show skipped entries",
                            key=(
                                f"zip_skipped_{chat_id}_"
                                f"{version}_{zip_hash}"
                            ),
                            disabled=busy,
                        ):
                            st.code(
                                "\n".join(skipped[:100]),
                                language="text",
                            )

                            if len(skipped) > 100:
                                st.caption(
                                    "Showing the first 100 skipped entries."
                                )

                except AttachmentError as exc:
                    st.session_state.pop(cache_key, None)
                    st.error(str(exc))

            if st.button(
                "Remove draft upload",
                key=f"remove_zip_{chat_id}_{version}",
                disabled=busy,
            ):
                reset_attachment_draft(chat_id)
                st.rerun()

        else:
            st.session_state.pop(cache_key, None)

        st.caption(
            "Selections apply to new questions. Retry uses the original "
            "question's saved files. Removing a draft upload does not erase "
            "attachments already saved in conversation history."
        )

        st.caption(
            "For follow-up questions, avoid reattaching the same files if "
            "their original message is still included in conversation history."
        )

    st.session_state[selection_state_key] = selected_files
    return selected_files


# -------------------------
# Load persistent state
# -------------------------
chats = storage.list_chats()

if not chats:
    storage.create_chat()
    chats = storage.list_chats()

valid_ids = {item["id"] for item in chats}

if st.session_state.get("active_chat_id") not in valid_ids:
    st.session_state.active_chat_id = chats[0]["id"]

if "pending_job" not in st.session_state:
    st.session_state.pending_job = None

chat = storage.get_chat(st.session_state.active_chat_id)

# Another browser tab may have deleted the conversation.
if chat is None:
    st.session_state.pop("active_chat_id", None)
    st.session_state.pending_job = None
    st.rerun()

attempt = storage.latest_attempt(chat["id"])

running = bool(attempt and attempt["status"] == "running")
busy = running or st.session_state.pending_job is not None

unanswered = bool(
    chat["messages"] and chat["messages"][-1]["role"] == "user"
)


# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.image(str(ICON_PATH), width=56)
    st.title("Velora")
    st.caption("Local SQLite workspace")

    if st.button(
        "＋ New conversation",
        type="primary",
        use_container_width=True,
        disabled=busy,
    ):
        st.session_state.active_chat_id = storage.create_chat()
        st.rerun()

    st.divider()
    st.caption("CONVERSATIONS")

    for item in chats:
        selected = item["id"] == chat["id"]

        if st.button(
            f"{'●' if selected else '○'} {item['title']}",
            key=f"chat_{item['id']}",
            use_container_width=True,
            disabled=busy,
        ):
            st.session_state.active_chat_id = item["id"]
            st.rerun()

    st.divider()

    usage = storage.usage_today()

    st.subheader("Usage today")
    st.caption("Resets at midnight UTC")

    st.write(
        f"**Requests:** {usage['requests']:,} / "
        f"{limits['daily_requests']:,}"
    )
    st.progress(
        min(usage["requests"] / limits["daily_requests"], 1.0)
    )

    st.write(
        f"**Accounted tokens:** {usage['accounted_tokens']:,} / "
        f"{limits['daily_token_budget']:,}"
    )
    st.progress(
        min(
            usage["accounted_tokens"] / limits["daily_token_budget"],
            1.0,
        )
    )

    with st.expander("Usage details"):
        st.write(
            f"API-reported tokens: {usage['reported_tokens']:,}"
        )
        st.write(
            "Unconfirmed reservations: "
            f"{usage['unconfirmed_tokens']:,}"
        )
        st.write(
            f"Running requests today: {usage['running']}"
        )
        st.caption(
            "Accounted tokens include reservations for running requests "
            "and requests without final API usage. This is not an Azure bill."
        )

    with st.expander("⚙️ Model settings"):
        st.caption(f"Deployment: {deployment}")

        history_turns = st.slider(
            "Previous turns to include",
            min_value=0,
            max_value=30,
            value=10,
            key="history_turns",
            disabled=busy,
            help=(
                "Earlier attachments are included only when their messages "
                "are within this history window."
            ),
        )

        system_prompt = st.text_area(
            "System instructions",
            value=DEFAULT_SYSTEM_PROMPT,
            height=180,
            key="system_prompt",
            disabled=busy,
        )

        st.caption(
            f"Input estimate limit: {limits['max_input_tokens']:,}\n\n"
            f"Output token limit: {limits['max_output_tokens']:,}\n\n"
            "Hard limits are configured in secrets or environment variables."
        )

    with st.expander("🗑️ Conversation management"):
        st.caption(
            "Clearing or deleting a conversation removes its saved messages "
            "and attachment snapshots from active records. "
            "Usage accounting is retained."
        )

        if st.button(
            "Clear conversation",
            use_container_width=True,
            disabled=busy,
        ):
            try:
                storage.replace_chat(
                    chat,
                    [],
                    "New conversation",
                )
                reset_attachment_draft(chat["id"])
            except storage.ConflictError as exc:
                show_notice(str(exc))

            st.rerun()

        if st.button(
            "Delete conversation",
            use_container_width=True,
            disabled=busy,
        ):
            try:
                storage.delete_chat(chat)
                reset_attachment_draft(chat["id"])
                st.session_state.pop("active_chat_id", None)
            except storage.ConflictError as exc:
                show_notice(str(exc))

            st.rerun()

    st.caption(
        "History and admitted attachment snapshots are saved locally. "
        "All sessions share this database. Do not expose this app publicly "
        "without authentication."
    )


# -------------------------
# Conversation UI
# -------------------------
st.title("Velora Chat")
st.caption(
    "Persistent conversations · ZIP source context · "
    "Streaming responses · Usage controls"
)

notice = st.session_state.pop("notice", None)

if notice:
    st.warning(notice)

# Render before suggestion buttons and chat input so queue_request()
# captures the current attachment selection.
selected_files = render_zip_attachments(chat, busy)

if not chat["messages"]:
    st.subheader("What would you like to build?")

    if selected_files:
        suggestions = [
            (
                "🔎 Review selected files",
                "Review the attached files for bugs, maintainability issues, "
                "and potential security problems. Cite filenames and line "
                "numbers, and distinguish definite findings from hypotheses.",
            ),
            (
                "📖 Explain this project",
                "Explain how the attached files work together. Describe the "
                "main components and execution flow. Identify any missing "
                "files needed for a more complete explanation.",
            ),
        ]
    else:
        suggestions = [
            (
                "🐍 Write Python",
                "Write a Python function that removes duplicates from a list "
                "while preserving order. Include tests.",
            ),
            (
                "🔎 Explain code",
                "Explain Python async and await with a runnable example.",
            ),
        ]

    columns = st.columns(2)

    for index, (label, text) in enumerate(suggestions):
        with columns[index]:
            if st.button(
                label,
                key=f"suggestion_{index}",
                use_container_width=True,
                disabled=busy,
            ):
                queue_request(chat, text)
                st.rerun()

for message in chat["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_saved_attachments(message)

        metrics = message.get("metrics", {})

        if metrics:
            st.caption(metrics_summary(metrics))

            with st.expander("Response details"):
                st.caption(
                    f"Deployment: {metrics.get('deployment', 'Unknown')}"
                )

                first_text = metrics.get("time_to_first_text_seconds")

                if first_text is not None:
                    st.caption(
                        f"Time to first text: {first_text:.2f} s"
                    )

                st.caption(
                    "API token counts include conversation and attachment "
                    "context sent with this request."
                )

if attempt and attempt["status"] in {"failed", "interrupted"}:
    st.warning(f"Latest attempt: {attempt['error']}")

if running:
    st.info(
        "A request is running for this conversation. If it finished in "
        "another tab, refresh the page. If the server stopped during generation, "
        "stop Streamlit and run `python storage.py --recover` before restarting."
    )

if chat["messages"]:
    retry_col, export_col, discard_col = st.columns(3)

    with retry_col:
        if st.button(
            "↻ Retry" if unanswered else "↻ Regenerate",
            use_container_width=True,
            disabled=busy,
            help=(
                "Uses the original question's saved attachments and the "
                "current history/system settings."
            ),
        ):
            queue_request(chat)
            st.rerun()

    with export_col:
        st.download_button(
            "↓ Export",
            data=export_markdown(chat),
            file_name=f"chat-{chat['id'][:8]}.md",
            mime="text/markdown",
            use_container_width=True,
            disabled=busy,
        )

    with discard_col:
        if unanswered and st.button(
            "Discard question",
            use_container_width=True,
            disabled=busy,
        ):
            remaining = chat["messages"][:-1]

            try:
                storage.replace_chat(
                    chat,
                    remaining,
                    (
                        chat["title"]
                        if remaining
                        else "New conversation"
                    ),
                )
            except storage.ConflictError as exc:
                show_notice(str(exc))

            st.rerun()

if unanswered and not busy:
    st.caption(
        "Retry or discard the unanswered question before continuing. "
        "Changing the draft upload does not change attachments on that question."
    )

if selected_files and not busy and not unanswered:
    st.caption(
        f"📎 {len(selected_files)} selected file(s) will be attached "
        "to your next question."
    )

prompt = st.chat_input(
    "Ask about your code or message Velora Chat…",
    disabled=busy or unanswered,
)

if prompt and prompt.strip():
    queue_request(chat, prompt.strip())
    st.rerun()


# -------------------------
# Execute queued request
# -------------------------
job = st.session_state.pending_job

if job is not None:
    # Consume the UI job before making a network request.
    # An interrupted rerun must not automatically resend it.
    st.session_state.pending_job = None

    current = storage.get_chat(job["chat_id"])

    if current is None or current["revision"] != job["revision"]:
        show_notice(
            "Conversation changed. Please refresh and try again."
        )
        st.rerun()

    new_prompt = job["prompt"]

    if new_prompt is not None:
        # Guard against adding another question after an unanswered one.
        if (
            current["messages"]
            and current["messages"][-1]["role"] == "user"
        ):
            show_notice(
                "Retry or discard the unanswered question first."
            )
            st.rerun()

        user_message = {
            "role": "user",
            "content": new_prompt,
        }

        selected_attachments = job.get("attachments", [])

        if selected_attachments:
            user_message["attachments"] = selected_attachments

        base_messages = [
            *current["messages"],
            user_message,
        ]
        pending_messages = base_messages

        title = current["title"]

        if not current["messages"]:
            compact = " ".join(new_prompt.split())
            title = compact[:48] + (
                "…" if len(compact) > 48 else ""
            )

    else:
        last_user = next(
            (
                index
                for index in range(
                    len(current["messages"]) - 1,
                    -1,
                    -1,
                )
                if current["messages"][index]["role"] == "user"
            ),
            None,
        )

        if last_user is None:
            show_notice("There is no question to retry.")
            st.rerun()

        # Saved attachments remain on the original user message.
        base_messages = current["messages"][: last_user + 1]

        # Keep the old successful answer until regeneration succeeds.
        pending_messages = current["messages"]
        title = current["title"]

    previous = base_messages[:-1]
    previous = (
        previous[-history_turns * 2 :]
        if history_turns
        else []
    )

    # Convert persisted messages into supported API fields.
    # User attachment snapshots become text inside their message content.
    request_messages = [
        {
            "role": "system",
            "content": (
                (system_prompt.strip() or DEFAULT_SYSTEM_PROMPT)
                + "\n\n"
                + ATTACHMENT_INSTRUCTIONS
            ),
        },
        *[
            to_api_message(item)
            for item in [
                *previous,
                base_messages[-1],
            ]
        ],
    ]

    # Includes filenames, numbered source text, history, instructions,
    # and the current question before reserving usage.
    estimated_input = estimate_input_tokens(request_messages)

    try:
        attempt_id = storage.reserve_attempt(
            chat=current,
            pending_messages=pending_messages,
            title=title,
            deployment=deployment,
            estimated_input=estimated_input,
            limits=limits,
        )
    except (storage.LimitError, storage.ConflictError) as exc:
        show_notice(str(exc))
        st.rerun()

    if new_prompt is not None:
        with st.chat_message("user"):
            st.markdown(new_prompt)
            render_saved_attachments(base_messages[-1])

    metrics = {}
    error = None
    completed_messages = None

    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(
                stream_answer(request_messages, metrics)
            )

            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError(
                    "The deployment returned no text."
                )

            completed_messages = [
                *base_messages,
                {
                    "role": "assistant",
                    "content": answer,
                    "metrics": metrics.copy(),
                },
            ]

        except Exception as exc:
            error = friendly_error(exc)

            logger.warning(
                "Request failed: type=%s status=%s request_id=%s",
                type(exc).__name__,
                getattr(exc, "status_code", None),
                getattr(exc, "request_id", None),
            )

    # If execution is forcibly interrupted before this point, the attempt
    # remains running and its reservation remains charged.
    try:
        storage.finish_attempt(
            attempt_id,
            metrics,
            error=error,
            completed_messages=completed_messages,
        )
    except Exception:
        logger.exception("Could not persist request completion.")

        st.error(
            "The request finished, but its result could not be saved. "
            "Its reservation remains in place. Check the database and logs."
        )
        st.stop()

    st.rerun()