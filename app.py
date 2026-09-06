import logging
import os
import time

import streamlit as st
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

import storage


st.set_page_config(
    page_title="Foundry Chat",
    page_icon="💬",
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
        max-width: 900px;
        padding-top: 2rem;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(255,255,255,0.07);
    }
    [data-testid="stChatMessage"] {
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        margin-bottom: 1rem;
    }
    div.stButton > button,
    div.stDownloadButton > button {
        border-radius: 10px;
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
    str(get_setting("AZURE_OPENAI_INCLUDE_USAGE", True)).lower()
    in {"true", "1", "yes", "on"}
)

output_parameter = str(
    get_setting(
        "AZURE_OPENAI_OUTPUT_PARAMETER",
        "max_completion_tokens",
    )
)

if not all([endpoint, api_key, deployment]):
    st.error("Configure the Azure OpenAI endpoint, API key, and deployment.")
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
        return "Could not reach the deployment. Check your network and endpoint."

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
            # Usage can arrive with no choices.
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
    st.session_state.pending_job = {
        "chat_id": chat["id"],
        "revision": chat["revision"],
        "prompt": prompt,
    }


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
attempt = storage.latest_attempt(chat["id"])

running = bool(attempt and attempt["status"] == "running")
busy = running or st.session_state.pending_job is not None


# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.title("💬 Foundry Chat")
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
        st.write(f"API-reported tokens: {usage['reported_tokens']:,}")
        st.write(
            "Unconfirmed reservations: "
            f"{usage['unconfirmed_tokens']:,}"
        )
        st.write(f"Running requests today: {usage['running']}")
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
        st.caption("Deletion does not reset usage accounting.")

        if st.button(
            "Clear conversation",
            use_container_width=True,
            disabled=busy,
        ):
            try:
                storage.replace_chat(chat, [], "New conversation")
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
                st.session_state.pop("active_chat_id", None)
            except storage.ConflictError as exc:
                show_notice(str(exc))
            st.rerun()

    st.caption(
        "History is saved locally. All sessions share this database. "
        "Do not expose this app publicly without authentication."
    )


# -------------------------
# Conversation UI
# -------------------------
st.title("Foundry Chat")
st.caption("Persistent conversations · Streaming responses · Usage controls")

notice = st.session_state.pop("notice", None)
if notice:
    st.warning(notice)

if not chat["messages"]:
    st.subheader("What would you like to build?")

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
                use_container_width=True,
                disabled=busy,
            ):
                queue_request(chat, text)
                st.rerun()

for message in chat["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        metrics = message.get("metrics", {})
        if metrics:
            st.caption(metrics_summary(metrics))

            with st.expander("Response details"):
                st.caption(
                    f"Deployment: {metrics.get('deployment', 'Unknown')}"
                )
                first_text = metrics.get("time_to_first_text_seconds")

                if first_text is not None:
                    st.caption(f"Time to first text: {first_text:.2f} s")

                st.caption(
                    "API token counts include the context sent with this request."
                )

if attempt and attempt["status"] in {"failed", "interrupted"}:
    st.warning(f"Latest attempt: {attempt['error']}")

if running:
    st.info(
        "A request is running for this conversation. If it finished in "
        "another tab, refresh. If the server stopped during generation, "
        "use the recovery command described below."
    )

unanswered = bool(
    chat["messages"] and chat["messages"][-1]["role"] == "user"
)

if chat["messages"]:
    retry_col, export_col, discard_col = st.columns(3)

    with retry_col:
        if st.button(
            "↻ Retry" if unanswered else "↻ Regenerate",
            use_container_width=True,
            disabled=busy,
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
                    chat["title"] if remaining else "New conversation",
                )
            except storage.ConflictError as exc:
                show_notice(str(exc))
            st.rerun()

if unanswered and not busy:
    st.caption("Retry or discard the unanswered question before continuing.")

prompt = st.chat_input(
    "Message Foundry Chat…",
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
        show_notice("Conversation changed. Please refresh and try again.")
        st.rerun()

    new_prompt = job["prompt"]

    if new_prompt is not None:
        base_messages = [
            *current["messages"],
            {"role": "user", "content": new_prompt},
        ]
        pending_messages = base_messages

        title = current["title"]
        if not current["messages"]:
            compact = " ".join(new_prompt.split())
            title = compact[:48] + ("…" if len(compact) > 48 else "")
    else:
        last_user = next(
            (
                index
                for index in range(len(current["messages"]) - 1, -1, -1)
                if current["messages"][index]["role"] == "user"
            ),
            None,
        )

        if last_user is None:
            show_notice("There is no question to retry.")
            st.rerun()

        base_messages = current["messages"][: last_user + 1]

        # Keep the old successful answer until regeneration succeeds.
        pending_messages = current["messages"]
        title = current["title"]

    previous = base_messages[:-1]
    previous = previous[-history_turns * 2 :] if history_turns else []

    request_messages = [
        {
            "role": "system",
            "content": system_prompt.strip() or DEFAULT_SYSTEM_PROMPT,
        },
        *[
            {"role": item["role"], "content": item["content"]}
            for item in [*previous, base_messages[-1]]
        ],
    ]

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

    metrics = {}
    error = None
    completed_messages = None

    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(
                stream_answer(request_messages, metrics)
            )

            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("The deployment returned no text.")

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
    # remains 'running' and its reservation remains charged.
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