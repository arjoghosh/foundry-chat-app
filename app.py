import logging
import os
import uuid
from datetime import datetime, timezone

import streamlit as st
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

st.set_page_config(
    page_title="Foundry Chat",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
)

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """
You are a helpful, accurate AI assistant.

Use Markdown to make your answers easy to read.
Always put code in fenced code blocks with the appropriate language tag.
Keep explanations outside code blocks.
Provide complete, runnable examples when appropriate.
Be clear about uncertainty and do not invent facts.
""".strip()

# Cosmetic styling. The actual dark theme comes from config.toml.
st.markdown(
    """
    <style>
    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(255,255,255,0.07);
    }

    [data-testid="stChatMessage"] {
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        margin-bottom: 1rem;
    }

    div.stButton > button {
        border-radius: 10px;
    }

    div.stDownloadButton > button {
        border-radius: 10px;
    }

    [data-testid="stChatInput"] {
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------
# Configuration and client
# -------------------------
def get_setting(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value:
        return value

    try:
        return str(st.secrets.get(name, default))
    except FileNotFoundError:
        return default


@st.cache_resource
def create_client(endpoint: str, api_key: str) -> OpenAI:
    return OpenAI(
        base_url=endpoint.rstrip("/") + "/",
        api_key=api_key,
        timeout=90.0,
        max_retries=2,
    )


endpoint = get_setting("AZURE_OPENAI_ENDPOINT")
api_key = get_setting("AZURE_OPENAI_API_KEY")
deployment = get_setting("AZURE_OPENAI_DEPLOYMENT")

if not all([endpoint, api_key, deployment]):
    st.error(
        "Missing configuration. Set AZURE_OPENAI_ENDPOINT, "
        "AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT "
        "in Streamlit secrets or environment variables."
    )
    st.stop()

client = create_client(endpoint, api_key)


# -------------------------
# Session state
# -------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_chat() -> dict:
    return {
        "title": "New conversation",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "messages": [],
        "error": None,
    }


if "chats" not in st.session_state:
    first_id = uuid.uuid4().hex
    st.session_state.chats = {first_id: make_chat()}
    st.session_state.active_chat_id = first_id

if "pending_job" not in st.session_state:
    st.session_state.pending_job = None


def active_chat() -> dict:
    return st.session_state.chats[st.session_state.active_chat_id]


def new_chat():
    chat_id = uuid.uuid4().hex
    st.session_state.chats[chat_id] = make_chat()
    st.session_state.active_chat_id = chat_id
    st.session_state.pending_job = None


def select_chat(chat_id: str):
    st.session_state.active_chat_id = chat_id
    st.session_state.pending_job = None


def clear_chat():
    chat_id = st.session_state.active_chat_id
    st.session_state.chats[chat_id] = make_chat()
    st.session_state.pending_job = None


def delete_chat():
    chat_id = st.session_state.active_chat_id
    del st.session_state.chats[chat_id]
    st.session_state.pending_job = None

    if not st.session_state.chats:
        new_chat()
        return

    st.session_state.active_chat_id = max(
        st.session_state.chats,
        key=lambda key: st.session_state.chats[key]["updated_at"],
    )


def submit_prompt(prompt: str):
    prompt = prompt.strip()
    if not prompt:
        return

    chat = active_chat()

    # Keep failed questions in history, but resolve the latest one
    # before accepting a new question.
    if chat["messages"] and chat["messages"][-1]["role"] == "user":
        return

    if not chat["messages"]:
        title = " ".join(prompt.split())
        chat["title"] = title[:48] + ("…" if len(title) > 48 else "")

    chat["messages"].append({"role": "user", "content": prompt})
    chat["error"] = None
    chat["updated_at"] = now_iso()

    st.session_state.pending_job = {
        "chat_id": st.session_state.active_chat_id,
        "user_index": len(chat["messages"]) - 1,
    }


def retry_latest():
    chat = active_chat()

    last_user_index = next(
        (
            index
            for index in range(len(chat["messages"]) - 1, -1, -1)
            if chat["messages"][index]["role"] == "user"
        ),
        None,
    )

    if last_user_index is None:
        return

    # Replace the latest answer instead of duplicating the question.
    chat["messages"] = chat["messages"][: last_user_index + 1]
    chat["error"] = None
    chat["updated_at"] = now_iso()

    st.session_state.pending_job = {
        "chat_id": st.session_state.active_chat_id,
        "user_index": last_user_index,
    }


def discard_unanswered():
    chat = active_chat()

    if chat["messages"] and chat["messages"][-1]["role"] == "user":
        chat["messages"].pop()

    chat["error"] = None
    chat["updated_at"] = now_iso()
    st.session_state.pending_job = None

    if not chat["messages"]:
        chat["title"] = "New conversation"


def export_markdown(chat: dict) -> str:
    sections = [f"# {chat['title']}", ""]

    for message in chat["messages"]:
        speaker = "You" if message["role"] == "user" else "Assistant"
        sections.append(f"## {speaker}\n\n{message['content']}\n")

    return "\n".join(sections)


# -------------------------
# Model streaming
# -------------------------
def stream_answer(messages: list):
    stream = client.chat.completions.create(
        model=deployment,
        messages=messages,
        stream=True,
    )

    try:
        for chunk in stream:
            if not chunk.choices:
                continue

            choice = chunk.choices[0]

            if choice.finish_reason == "content_filter":
                raise RuntimeError(
                    "The response was stopped by the deployment's safety filter. "
                    "Try rephrasing your question."
                )

            if choice.delta.content:
                yield choice.delta.content

            if choice.finish_reason == "length":
                yield (
                    "\n\n---\n"
                    "*The response reached the model's output limit. "
                    "Ask me to continue if needed.*"
                )
    finally:
        stream.close()


def friendly_error(exc: Exception) -> str:
    if isinstance(exc, AuthenticationError):
        return "Authentication failed. Check the resource API key and endpoint."

    if isinstance(exc, RateLimitError):
        return "The deployment is rate-limited. Wait a moment, then retry."

    if isinstance(exc, APIConnectionError):
        return "Could not reach the deployment. Check your network and endpoint."

    if isinstance(exc, APIStatusError):
        if exc.status_code == 404:
            return (
                "Deployment or API route not found. Check the exact deployment "
                "name, the /openai/v1/ endpoint, and Chat Completions support."
            )

        if exc.status_code == 400:
            return (
                "The deployment rejected the request. Check model compatibility, "
                "content restrictions, and conversation length."
            )

        return f"The deployment returned HTTP {exc.status_code}. Please retry."

    if isinstance(exc, RuntimeError):
        return str(exc)

    return "An unexpected error occurred. Please retry."


# -------------------------
# Sidebar
# -------------------------
busy = st.session_state.pending_job is not None

with st.sidebar:
    st.title("💬 Foundry Chat")
    st.caption("Your AI coding workspace")

    st.button(
        "＋ New conversation",
        type="primary",
        use_container_width=True,
        on_click=new_chat,
        disabled=busy,
    )

    st.divider()
    st.caption("CONVERSATIONS")

    ordered_chats = sorted(
        st.session_state.chats.items(),
        key=lambda item: item[1]["updated_at"],
        reverse=True,
    )

    for chat_id, item in ordered_chats:
        selected = chat_id == st.session_state.active_chat_id

        st.button(
            f"{'●' if selected else '○'} {item['title']}",
            key=f"chat_{chat_id}",
            use_container_width=True,
            on_click=select_chat,
            args=(chat_id,),
            disabled=busy,
        )

    st.divider()

    with st.expander("⚙️ Model settings"):
        st.caption(f"Deployment: {deployment}")

        history_turns = st.slider(
            "Previous turns to include",
            min_value=0,
            max_value=30,
            value=10,
            disabled=busy,
            help=(
                "A turn is one question and answer. This is not a token limit; "
                "large messages can still exceed the model's context window."
            ),
        )

        system_prompt = st.text_area(
            "System instructions",
            value=DEFAULT_SYSTEM_PROMPT,
            height=190,
            disabled=busy,
            help="Applies to the next response, including regenerated responses.",
        )

    with st.expander("🗑️ Conversation management"):
        st.caption("These actions cannot be undone.")

        st.button(
            "Clear current conversation",
            use_container_width=True,
            on_click=clear_chat,
            disabled=busy,
        )

        st.button(
            "Delete current conversation",
            use_container_width=True,
            on_click=delete_chat,
            disabled=busy,
        )

    st.caption(
        "History is stored in this session only. "
        "Hover over a code block to copy its contents."
    )


# -------------------------
# Conversation interface
# -------------------------
chat = active_chat()

st.title("Foundry Chat")
st.caption("Ask questions, build ideas, and write code.")

if not chat["messages"]:
    st.subheader("What would you like to work on?")

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
        (
            "🛠️ Design an API",
            "Create a small FastAPI CRUD API with input validation.",
        ),
        (
            "🧠 Learn something",
            "Explain retrieval-augmented generation with a practical example.",
        ),
    ]

    columns = st.columns(2)

    for index, (label, suggestion) in enumerate(suggestions):
        with columns[index % 2]:
            st.button(
                label,
                key=f"suggestion_{index}",
                use_container_width=True,
                on_click=submit_prompt,
                args=(suggestion,),
                disabled=busy,
            )

for message in chat["messages"]:
    avatar = "🧑‍💻" if message["role"] == "user" else "🤖"

    with st.chat_message(message["role"], avatar=avatar):
        # Markdown fenced code blocks include native copy controls.
        st.markdown(message["content"])

if chat["error"]:
    st.error(chat["error"])

unanswered = bool(
    chat["messages"] and chat["messages"][-1]["role"] == "user"
)

if chat["messages"]:
    retry_col, export_col, discard_col = st.columns([1, 1, 1])

    with retry_col:
        st.button(
            "↻ Retry response" if unanswered else "↻ Regenerate",
            use_container_width=True,
            on_click=retry_latest,
            disabled=busy,
            help="Replaces the latest answer using the current settings.",
        )

    with export_col:
        st.download_button(
            "↓ Export chat",
            data=export_markdown(chat),
            file_name=(
                f"foundry-chat-{st.session_state.active_chat_id[:8]}.md"
            ),
            mime="text/markdown",
            use_container_width=True,
            disabled=busy,
        )

    with discard_col:
        if unanswered:
            st.button(
                "Discard question",
                use_container_width=True,
                on_click=discard_unanswered,
                disabled=busy,
            )

if unanswered and not busy:
    st.caption(
        "Retry the unanswered question, or discard it to send a new message."
    )

prompt = st.chat_input(
    "Message Foundry Chat…",
    disabled=busy or unanswered,
)

if prompt:
    submit_prompt(prompt)
    st.rerun()


# -------------------------
# Execute a queued request
# -------------------------
job = st.session_state.pending_job

if job is not None:
    target_chat = st.session_state.chats[job["chat_id"]]
    user_index = job["user_index"]

    # Include previous complete turns plus the current question.
    previous = target_chat["messages"][:user_index]
    previous = previous[-history_turns * 2 :] if history_turns else []

    request_messages = [
        {
            "role": "system",
            "content": system_prompt.strip() or DEFAULT_SYSTEM_PROMPT,
        },
        *previous,
        target_chat["messages"][user_index],
    ]

    with st.chat_message("assistant", avatar="🤖"):
        try:
            answer = st.write_stream(stream_answer(request_messages))

            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError(
                    "The deployment returned no text. Retry or verify that "
                    "the deployment supports text Chat Completions."
                )

            target_chat["messages"].append(
                {"role": "assistant", "content": answer}
            )
            target_chat["error"] = None

        except Exception as exc:
            # Do not save partial failed responses as complete answers.
            # Keep raw exception details out of the public interface.
            logger.warning(
                "Chat request failed: type=%s status=%s request_id=%s",
                type(exc).__name__,
                getattr(exc, "status_code", None),
                getattr(exc, "request_id", None),
            )
            target_chat["error"] = friendly_error(exc)

        finally:
            target_chat["updated_at"] = now_iso()
            st.session_state.pending_job = None

    # Render the canonical saved history and restore the controls.
    st.rerun()