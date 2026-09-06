import os

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
)

SYSTEM_PROMPT = """
You are a helpful, accurate AI assistant.

Use Markdown to make your answers easy to read.
When providing code, always use fenced code blocks with a language tag.
Keep explanations outside code blocks.
Provide complete, runnable examples when appropriate.
Never put an entire answer inside a code block unless explicitly requested.
"""


def get_setting(name: str, default: str = "") -> str:
    """Read configuration from environment variables or Streamlit secrets."""
    value = os.getenv(name)
    if value:
        return value

    try:
        return str(st.secrets.get(name, default))
    except FileNotFoundError:
        return default


@st.cache_resource
def create_client(
    endpoint: str,
    api_key: str,
) -> OpenAI:
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
        "in .streamlit/secrets.toml or environment variables."
    )
    st.stop()

client = create_client(endpoint, api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


def reset_chat():
    st.session_state.messages = []
    st.session_state.pending_prompt = None


def queue_prompt(text: str):
    st.session_state.pending_prompt = text


def stream_answer(messages):
    """Yield text only; ignore metadata and empty streaming chunks."""
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
                    "The response was stopped by the deployment's safety filter."
                )

            content = choice.delta.content
            if content:
                yield content
    finally:
        stream.close()


# Sidebar
with st.sidebar:
    st.title("💬 Foundry Chat")

    st.button(
        "＋ New chat",
        use_container_width=True,
        on_click=reset_chat,
    )

    st.divider()
    st.caption(f"Deployment: {deployment}")

    history_turns = st.slider(
        "Previous conversation turns to send",
        min_value=0,
        max_value=30,
        value=10,
        help=(
            "Limits how many previous question-and-answer pairs are sent "
            "to the model. Large messages can still exceed its context limit."
        ),
    )

    st.divider()
    st.markdown(
        "**Copy code**\n\n"
        "Hover over a code block and click its copy icon."
    )


# Main chat
st.title("How can I help you?")
st.caption("Powered by Microsoft Foundry")

if not st.session_state.messages:
    col1, col2 = st.columns(2)

    with col1:
        st.button(
            "🐍 Write Python code",
            use_container_width=True,
            on_click=queue_prompt,
            args=(
                "Write a Python function that removes duplicates "
                "from a list while preserving order.",
            ),
        )

    with col2:
        st.button(
            "🔎 Explain some code",
            use_container_width=True,
            on_click=queue_prompt,
            args=(
                "Explain Python list comprehensions with runnable examples.",
            ),
        )

# Render saved messages. Streamlit renders fenced Markdown code blocks
# with syntax highlighting and native copy controls.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

typed_prompt = st.chat_input("Message Foundry Chat…")
prompt = typed_prompt or st.session_state.pending_prompt
st.session_state.pending_prompt = None

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stored history contains only successful, complete exchanges.
    previous_messages = (
        st.session_state.messages[-history_turns * 2 :]
        if history_turns
        else []
    )

    request_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *previous_messages,
        {"role": "user", "content": prompt},
    ]

    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(stream_answer(request_messages))

            if not isinstance(answer, str) or not answer.strip():
                st.warning("No text response was returned. Please try again.")
            else:
                st.session_state.messages.extend(
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": answer},
                    ]
                )

        except AuthenticationError:
            st.error(
                "Authentication failed. Check your Azure OpenAI API key "
                "and resource endpoint."
            )

        except RateLimitError:
            st.error(
                "The deployment is currently rate-limited. "
                "Wait a moment and try again."
            )

        except APIConnectionError:
            st.error(
                "Could not connect to Azure OpenAI. "
                "Check your endpoint and network access."
            )

        except APIStatusError as exc:
            st.error(
                f"The deployment returned HTTP {exc.status_code}. "
                "Check the deployment name, API version, model support, "
                "and input length."
            )

        except RuntimeError as exc:
            st.error(str(exc))

        except Exception:
            st.error("Something went wrong. Please try again.")