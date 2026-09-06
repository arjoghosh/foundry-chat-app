import json
import sqlite3
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "data"
DB_PATH = DATA_DIR / "chat.db"


class LimitError(Exception):
    pass


class ConflictError(Exception):
    pass


def utc_day() -> str:
    return datetime.now(timezone.utc).date().isoformat()


@contextmanager
def connection():
    db = sqlite3.connect(DB_PATH, timeout=15)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    db.execute("PRAGMA busy_timeout = 15000")

    try:
        yield db
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def initialize():
    DATA_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)

    with connection() as db:
        db.execute("PRAGMA journal_mode = WAL")

        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                messages_json TEXT NOT NULL DEFAULT '[]',
                revision INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS attempts (
                id TEXT PRIMARY KEY,
                conversation_id TEXT
                    REFERENCES conversations(id) ON DELETE SET NULL,
                day TEXT NOT NULL,
                deployment TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at REAL NOT NULL,
                completed_at REAL,
                reserved_tokens INTEGER NOT NULL,
                accounted_tokens INTEGER NOT NULL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                total_tokens INTEGER,
                usage_available INTEGER NOT NULL DEFAULT 0,
                duration_seconds REAL,
                first_text_seconds REAL,
                error TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_attempts_day
                ON attempts(day);

            CREATE INDEX IF NOT EXISTS idx_attempts_conversation
                ON attempts(conversation_id);

            CREATE UNIQUE INDEX IF NOT EXISTS idx_one_running_per_chat
                ON attempts(conversation_id)
                WHERE status = 'running';
            """
        )


def _decode_chat(row):
    if row is None:
        return None

    chat = dict(row)
    chat["messages"] = json.loads(chat.pop("messages_json"))
    return chat


def list_chats():
    with connection() as db:
        rows = db.execute(
            """
            SELECT id, title, updated_at
            FROM conversations
            ORDER BY updated_at DESC
            """
        ).fetchall()

    return [dict(row) for row in rows]


def get_chat(chat_id):
    with connection() as db:
        row = db.execute(
            "SELECT * FROM conversations WHERE id = ?",
            (chat_id,),
        ).fetchone()

    return _decode_chat(row)


def create_chat():
    chat_id = uuid.uuid4().hex
    now = time.time()

    with connection() as db:
        db.execute(
            """
            INSERT INTO conversations
                (id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (chat_id, "New conversation", now, now),
        )

    return chat_id


def _ensure_editable(db, chat_id, revision):
    row = db.execute(
        "SELECT revision FROM conversations WHERE id = ?",
        (chat_id,),
    ).fetchone()

    if row is None:
        raise ConflictError("This conversation was deleted. Refresh the app.")

    if row["revision"] != revision:
        raise ConflictError(
            "This conversation changed in another tab. Refresh and try again."
        )

    running = db.execute(
        """
        SELECT 1 FROM attempts
        WHERE conversation_id = ? AND status = 'running'
        """,
        (chat_id,),
    ).fetchone()

    if running:
        raise ConflictError(
            "This conversation already has a running request."
        )


def replace_chat(chat, messages, title=None):
    with connection() as db:
        db.execute("BEGIN IMMEDIATE")
        _ensure_editable(db, chat["id"], chat["revision"])

        db.execute(
            """
            UPDATE conversations
            SET messages_json = ?, title = ?, updated_at = ?,
                revision = revision + 1
            WHERE id = ?
            """,
            (
                json.dumps(messages),
                title if title is not None else chat["title"],
                time.time(),
                chat["id"],
            ),
        )


def delete_chat(chat):
    with connection() as db:
        db.execute("BEGIN IMMEDIATE")
        _ensure_editable(db, chat["id"], chat["revision"])

        # Attempts remain in the usage ledger. The foreign key becomes NULL.
        db.execute(
            "DELETE FROM conversations WHERE id = ?",
            (chat["id"],),
        )


def latest_attempt(chat_id):
    with connection() as db:
        row = db.execute(
            """
            SELECT * FROM attempts
            WHERE conversation_id = ?
            ORDER BY started_at DESC
            LIMIT 1
            """,
            (chat_id,),
        ).fetchone()

    return dict(row) if row else None


def usage_today():
    with connection() as db:
        row = db.execute(
            """
            SELECT
                COUNT(*) AS requests,
                COALESCE(SUM(accounted_tokens), 0) AS accounted_tokens,
                COALESCE(SUM(
                    CASE WHEN usage_available = 1
                    THEN total_tokens ELSE 0 END
                ), 0) AS reported_tokens,
                COALESCE(SUM(
                    CASE WHEN usage_available = 0
                    THEN accounted_tokens ELSE 0 END
                ), 0) AS unconfirmed_tokens,
                COALESCE(SUM(
                    CASE WHEN status = 'running' THEN 1 ELSE 0 END
                ), 0) AS running
            FROM attempts
            WHERE day = ?
            """,
            (utc_day(),),
        ).fetchone()

    return dict(row)


def reserve_attempt(
    chat,
    pending_messages,
    title,
    deployment,
    estimated_input,
    limits,
):
    """
    Atomically check limits, reserve budget, and save a new user question.

    Every admitted attempt counts toward the daily request limit, even if
    the provider subsequently rejects it.
    """
    if estimated_input > limits["max_input_tokens"]:
        raise LimitError(
            f"Estimated input is {estimated_input:,} tokens; "
            f"the configured limit is {limits['max_input_tokens']:,}. "
            "Reduce conversation history or shorten your question."
        )

    reservation = estimated_input + limits["max_output_tokens"]
    now = time.time()
    day = utc_day()
    attempt_id = uuid.uuid4().hex

    with connection() as db:
        # Serializes budget checks across browser tabs.
        db.execute("BEGIN IMMEDIATE")
        _ensure_editable(db, chat["id"], chat["revision"])

        totals = db.execute(
            """
            SELECT COUNT(*) AS requests,
                   COALESCE(SUM(accounted_tokens), 0) AS tokens
            FROM attempts
            WHERE day = ?
            """,
            (day,),
        ).fetchone()

        if totals["requests"] >= limits["daily_requests"]:
            raise LimitError(
                "Daily request limit reached. It resets at midnight UTC."
            )

        if totals["tokens"] + reservation > limits["daily_token_budget"]:
            remaining = max(
                0,
                limits["daily_token_budget"] - totals["tokens"],
            )
            raise LimitError(
                f"Insufficient daily token budget. "
                f"Remaining: {remaining:,}; "
                f"required reservation: {reservation:,}. "
                "Reduce context or wait until the next UTC day."
            )

        last_started = db.execute(
            "SELECT MAX(started_at) FROM attempts"
        ).fetchone()[0]

        if last_started is not None:
            wait = (
                limits["min_seconds_between_requests"]
                - (now - last_started)
            )

            if wait > 0:
                raise LimitError(
                    f"Please wait {wait:.1f} seconds before another request."
                )

        db.execute(
            """
            INSERT INTO attempts (
                id, conversation_id, day, deployment, status,
                started_at, reserved_tokens, accounted_tokens
            )
            VALUES (?, ?, ?, ?, 'running', ?, ?, ?)
            """,
            (
                attempt_id,
                chat["id"],
                day,
                deployment,
                now,
                reservation,
                reservation,
            ),
        )

        db.execute(
            """
            UPDATE conversations
            SET messages_json = ?, title = ?, updated_at = ?,
                revision = revision + 1
            WHERE id = ?
            """,
            (
                json.dumps(pending_messages),
                title,
                now,
                chat["id"],
            ),
        )

    return attempt_id


def finish_attempt(
    attempt_id,
    metrics,
    error=None,
    completed_messages=None,
):
    """
    Save attempt metrics and optionally replace the displayed conversation.

    Missing usage retains the reservation; it is never recorded as zero.
    """
    with connection() as db:
        db.execute("BEGIN IMMEDIATE")

        attempt = db.execute(
            "SELECT * FROM attempts WHERE id = ?",
            (attempt_id,),
        ).fetchone()

        if attempt is None or attempt["status"] != "running":
            raise ConflictError(
                "This request is no longer active. Its answer was not saved."
            )

        total = metrics.get("total_tokens")
        usage_available = (
            metrics.get("usage_available", False) and total is not None
        )

        accounted = (
            int(total)
            if usage_available
            else attempt["reserved_tokens"]
        )

        db.execute(
            """
            UPDATE attempts
            SET status = ?, completed_at = ?,
                accounted_tokens = ?,
                input_tokens = ?, output_tokens = ?, total_tokens = ?,
                usage_available = ?,
                duration_seconds = ?, first_text_seconds = ?,
                error = ?
            WHERE id = ?
            """,
            (
                "failed" if error else "completed",
                time.time(),
                accounted,
                metrics.get("input_tokens"),
                metrics.get("output_tokens"),
                total,
                int(usage_available),
                metrics.get("duration_seconds"),
                metrics.get("time_to_first_text_seconds"),
                error,
                attempt_id,
            ),
        )

        if completed_messages is not None:
            db.execute(
                """
                UPDATE conversations
                SET messages_json = ?, updated_at = ?,
                    revision = revision + 1
                WHERE id = ?
                """,
                (
                    json.dumps(completed_messages),
                    time.time(),
                    attempt["conversation_id"],
                ),
            )


def recover_interrupted():
    """
    Run only when Streamlit is stopped.

    Retain budget reservations for requests whose final usage is unknown.
    """
    with connection() as db:
        result = db.execute(
            """
            UPDATE attempts
            SET status = 'interrupted',
                completed_at = ?,
                error = 'Request interrupted; final usage is unknown.'
            WHERE status = 'running'
            """,
            (time.time(),),
        )

    return result.rowcount


if __name__ == "__main__":
    initialize()

    if "--recover" in sys.argv:
        count = recover_interrupted()
        print(f"Marked {count} request(s) interrupted. Reservations retained.")
    else:
        print(f"Database: {DB_PATH}")