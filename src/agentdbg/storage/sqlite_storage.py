"""
SQLite storage backend for traces.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

from agentdbg.models import Span, Trace

logger = logging.getLogger(__name__)


class SQLiteStorage:
    """SQLite-based storage for traces and spans."""

    def __init__(self, db_path: str = ".agentdbg/traces.db") -> None:
        self.db_path = db_path
        self._local = threading.local()
        self._ensure_db_exists()
        self._init_schema()

    def _ensure_db_exists(self) -> None:
        """Ensure the database directory exists."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            Path(db_dir).mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a thread-local database connection."""
        if not hasattr(self._local, "connection"):
            self._local.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
            )
            self._local.connection.row_factory = sqlite3.Row

        yield self._local.connection

    def _init_schema(self) -> None:
        """Initialize the database schema."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    name TEXT,
                    start_time REAL,
                    end_time REAL,
                    status TEXT,
                    total_cost REAL DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS spans (
                    span_id TEXT PRIMARY KEY,
                    trace_id TEXT,
                    parent_id TEXT,
                    name TEXT,
                    kind TEXT,
                    status TEXT,
                    start_time REAL,
                    end_time REAL,
                    duration_ms REAL,
                    input_data TEXT,
                    output_data TEXT,
                    attributes TEXT,
                    events TEXT,
                    cost_data TEXT,
                    error TEXT,
                    error_traceback TEXT,
                    FOREIGN KEY (trace_id) REFERENCES traces(trace_id)
                );

                CREATE INDEX IF NOT EXISTS idx_spans_trace_id ON spans(trace_id);
                CREATE INDEX IF NOT EXISTS idx_traces_start_time ON traces(start_time);
                CREATE INDEX IF NOT EXISTS idx_traces_status ON traces(status);
            """)
            conn.commit()

    def save_trace(self, trace: Trace) -> None:
        """Save a trace to the database."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO traces
                (trace_id, name, start_time, end_time, status, total_cost, total_tokens, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace.trace_id,
                    trace.name,
                    trace.start_time,
                    trace.end_time,
                    trace.status.value,
                    trace.total_cost,
                    trace.total_tokens,
                    json.dumps(trace.metadata),
                ),
            )
            conn.commit()

    def save_span(self, span: Span) -> None:
        """Save a span to the database."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO spans
                (span_id, trace_id, parent_id, name, kind, status, start_time, end_time,
                 duration_ms, input_data, output_data, attributes, events, cost_data,
                 error, error_traceback)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    span.span_id,
                    span.trace_id,
                    span.parent_id,
                    span.name,
                    span.kind.value,
                    span.status.value,
                    span.start_time,
                    span.end_time,
                    span.duration_ms,
                    json.dumps(span.input_data),
                    json.dumps(span.output_data),
                    json.dumps(span.attributes),
                    json.dumps(span.events),
                    json.dumps(span.cost.to_dict()),
                    span.error,
                    span.error_traceback,
                ),
            )
            conn.commit()

    def get_trace(self, trace_id: str) -> Trace | None:
        """Get a trace by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()

            if not row:
                return None

            trace = self._row_to_trace(row)

            # Load spans
            span_rows = conn.execute(
                "SELECT * FROM spans WHERE trace_id = ? ORDER BY start_time",
                (trace_id,),
            ).fetchall()

            for span_row in span_rows:
                trace.spans.append(self._row_to_span(span_row))

            return trace

    def get_traces(
        self,
        limit: int = 100,
        offset: int = 0,
        status: str | None = None,
    ) -> list[Trace]:
        """Get multiple traces."""
        with self._get_connection() as conn:
            query = "SELECT * FROM traces"
            params: list[Any] = []

            if status:
                query += " WHERE status = ?"
                params.append(status)

            query += " ORDER BY start_time DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            rows = conn.execute(query, params).fetchall()

            traces = []
            for row in rows:
                trace = self._row_to_trace(row)
                # Don't load spans by default for list view
                traces.append(trace)

            return traces

    def get_recent_traces(self, hours: int = 24) -> list[Trace]:
        """Get traces from the last N hours."""
        import time

        cutoff = time.time() - (hours * 3600)

        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM traces WHERE start_time > ? ORDER BY start_time DESC",
                (cutoff,),
            ).fetchall()

            return [self._row_to_trace(row) for row in rows]

    def delete_trace(self, trace_id: str) -> bool:
        """Delete a trace and its spans."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM spans WHERE trace_id = ?", (trace_id,))
            result = conn.execute("DELETE FROM traces WHERE trace_id = ?", (trace_id,))
            conn.commit()
            return result.rowcount > 0

    def delete_old_traces(self, days: int = 7) -> int:
        """Delete traces older than N days."""
        import time

        cutoff = time.time() - (days * 86400)

        with self._get_connection() as conn:
            # Get trace IDs to delete
            rows = conn.execute(
                "SELECT trace_id FROM traces WHERE start_time < ?", (cutoff,)
            ).fetchall()
            trace_ids = [row[0] for row in rows]

            if not trace_ids:
                return 0

            # Delete spans
            placeholders = ",".join("?" * len(trace_ids))
            conn.execute(
                f"DELETE FROM spans WHERE trace_id IN ({placeholders})", trace_ids
            )

            # Delete traces
            result = conn.execute(
                f"DELETE FROM traces WHERE trace_id IN ({placeholders})", trace_ids
            )
            conn.commit()

            return result.rowcount

    def get_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        with self._get_connection() as conn:
            trace_count = conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
            span_count = conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0]
            total_cost = (
                conn.execute("SELECT SUM(total_cost) FROM traces").fetchone()[0] or 0
            )
            total_tokens = (
                conn.execute("SELECT SUM(total_tokens) FROM traces").fetchone()[0] or 0
            )

            # Get error rate
            error_count = conn.execute(
                "SELECT COUNT(*) FROM traces WHERE status = 'error'"
            ).fetchone()[0]

            return {
                "trace_count": trace_count,
                "span_count": span_count,
                "total_cost": total_cost,
                "total_tokens": total_tokens,
                "error_count": error_count,
                "error_rate": error_count / trace_count if trace_count > 0 else 0,
            }

    def clear(self) -> None:
        """Clear all data."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM spans")
            conn.execute("DELETE FROM traces")
            conn.commit()

    def _row_to_trace(self, row: sqlite3.Row) -> Trace:
        """Convert a database row to a Trace object."""
        from agentdbg.models import SpanStatus

        return Trace(
            trace_id=row["trace_id"],
            name=row["name"] or "",
            start_time=row["start_time"],
            end_time=row["end_time"],
            status=SpanStatus(row["status"]),
            total_cost=row["total_cost"] or 0,
            total_tokens=row["total_tokens"] or 0,
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def _row_to_span(self, row: sqlite3.Row) -> Span:
        """Convert a database row to a Span object."""
        from agentdbg.models import CostInfo, SpanKind, SpanStatus

        cost_data = json.loads(row["cost_data"]) if row["cost_data"] else {}
        cost = CostInfo(
            input_tokens=cost_data.get("input_tokens", 0),
            output_tokens=cost_data.get("output_tokens", 0),
            total_tokens=cost_data.get("total_tokens", 0),
            input_cost=cost_data.get("input_cost", 0),
            output_cost=cost_data.get("output_cost", 0),
            total_cost=cost_data.get("total_cost", 0),
            model=cost_data.get("model", ""),
        )

        return Span(
            span_id=row["span_id"],
            trace_id=row["trace_id"],
            parent_id=row["parent_id"],
            name=row["name"] or "",
            kind=SpanKind(row["kind"]),
            status=SpanStatus(row["status"]),
            start_time=row["start_time"],
            end_time=row["end_time"],
            duration_ms=row["duration_ms"],
            input_data=json.loads(row["input_data"]) if row["input_data"] else {},
            output_data=json.loads(row["output_data"]) if row["output_data"] else {},
            attributes=json.loads(row["attributes"]) if row["attributes"] else {},
            events=json.loads(row["events"]) if row["events"] else [],
            cost=cost,
            error=row["error"],
            error_traceback=row["error_traceback"],
        )
