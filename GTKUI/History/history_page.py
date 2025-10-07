"""Embeddable history view for the ATLAS workspace."""

from __future__ import annotations

from typing import Any, Iterable, List

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

from GTKUI.Utils.utils import apply_css


class HistoryPage(Gtk.Box):
    """Display chat and tool history inside the main workspace."""

    def __init__(self, atlas: Any) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        self.ATLAS = atlas
        self.set_margin_top(16)
        self.set_margin_bottom(16)
        self.set_margin_start(16)
        self.set_margin_end(16)
        self.set_hexpand(True)
        self.set_vexpand(True)

        apply_css()

        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        header.set_halign(Gtk.Align.FILL)
        header.set_hexpand(True)
        self.append(header)

        title = Gtk.Label(label="History")
        title.set_xalign(0.0)
        title.get_style_context().add_class("title-2")
        header.append(title)

        header_spacer = Gtk.Box()
        header_spacer.set_hexpand(True)
        header.append(header_spacer)

        refresh_btn = Gtk.Button(label="Refresh")
        refresh_btn.set_tooltip_text("Reload chat and tool history from the ATLAS runtime.")
        refresh_btn.connect("clicked", lambda *_: self._refresh_contents())
        header.append(refresh_btn)

        description = Gtk.Label(
            label=(
                "Review the recent conversation, tool invocations, and internal events "
                "recorded by ATLAS."
            )
        )
        description.set_xalign(0.0)
        description.set_wrap(True)
        description.set_justify(Gtk.Justification.FILL)
        self.append(description)

        self._chat_view = self._create_text_section("Conversation history")
        self._tool_calls_view = self._create_text_section("Tool activity summary")
        self._tool_logs_view = self._create_text_section("Detailed tool logs")

        self._refresh_contents()

    # ------------------------------------------------------------------
    def _create_text_section(self, title: str) -> Gtk.TextView:
        frame = Gtk.Frame(label=title)
        frame.get_style_context().add_class("card")
        frame.set_hexpand(True)
        frame.set_vexpand(True)
        self.append(frame)

        scroller = Gtk.ScrolledWindow()
        scroller.set_hexpand(True)
        scroller.set_vexpand(True)
        scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        frame.set_child(scroller)

        view = Gtk.TextView()
        view.set_editable(False)
        view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        view.get_style_context().add_class("monospace")
        scroller.set_child(view)
        return view

    def _refresh_contents(self) -> None:
        chat_history = self._safe_call("get_chat_history_snapshot")
        formatted_chat = self._format_chat_history(chat_history)
        self._set_text(self._chat_view, formatted_chat)

        tool_entries = self._safe_call("get_tool_activity_log")
        formatted_tool_calls = self._format_tool_calls(tool_entries)
        self._set_text(self._tool_calls_view, formatted_tool_calls)

        formatted_tool_logs = self._format_tool_logs(tool_entries)
        self._set_text(self._tool_logs_view, formatted_tool_logs)

    def _safe_call(self, attribute: str) -> Any:
        getter = getattr(self.ATLAS, attribute, None)
        if not callable(getter):
            return None
        try:
            return getter()
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Formatting helpers (mirrors ChatPage terminal helpers in a compact form)
    # ------------------------------------------------------------------
    def _format_chat_history(self, entries: Any) -> str:
        if not isinstance(entries, Iterable):
            return "No conversation messages recorded yet."

        lines: List[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            timestamp = entry.get("timestamp") or ""
            role = entry.get("role") or "unknown"
            lines.append(f"[{timestamp}] {role}")

            content = self._stringify(entry.get("content"))
            if content:
                for segment in content.splitlines() or [""]:
                    lines.append(f"  {segment}")

            metadata = entry.get("metadata")
            metadata_text = self._stringify(metadata)
            if metadata_text:
                lines.append(f"  metadata: {metadata_text}")

            extra_keys = [
                key
                for key in entry.keys()
                if key not in {"timestamp", "role", "content", "metadata"}
            ]
            for key in extra_keys:
                value_text = self._stringify(entry.get(key))
                if value_text:
                    lines.append(f"  {key}: {value_text}")

            lines.append("")

        text = "\n".join(lines).strip()
        return text or "No conversation messages recorded yet."

    def _format_tool_calls(self, entries: Any) -> str:
        if not isinstance(entries, Iterable):
            return "No tool activity recorded."

        lines: List[str] = []
        for entry in reversed(list(entries)):
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("tool_name", "unknown"))
            timestamp = (
                entry.get("completed_at")
                or entry.get("started_at")
                or "Unknown time"
            )
            status = str(entry.get("status", "unknown")).upper()
            duration = entry.get("duration_ms")
            if isinstance(duration, (int, float)):
                duration_text = f"{duration:.0f} ms"
            else:
                duration_text = ""

            args_text = entry.get("arguments_text") or self._stringify(entry.get("arguments"))
            if args_text:
                args_text = " ".join(str(args_text).split())
                if len(args_text) > 120:
                    args_text = args_text[:117] + "..."

            line = f"[{timestamp}] {name}"
            if args_text:
                line += f"({args_text})"
            line += f" → {status}"
            if duration_text:
                line += f" • {duration_text}"
            lines.append(line)

        text = "\n".join(lines).strip()
        return text or "No tool activity recorded."

    def _format_tool_logs(self, entries: Any) -> str:
        if not isinstance(entries, Iterable):
            return "No tool logs available."

        blocks: List[str] = []
        for entry in reversed(list(entries)):
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("tool_name", "unknown"))
            status = str(entry.get("status", "unknown")).upper()
            started = entry.get("started_at") or "Unknown start"
            completed = entry.get("completed_at") or "Unknown end"
            duration = entry.get("duration_ms")
            if isinstance(duration, (int, float)):
                duration_line = f"Duration: {duration:.0f} ms"
            else:
                duration_line = None

            args_text = entry.get("arguments_text") or self._stringify(entry.get("arguments"))
            result_text = entry.get("result_text") or self._stringify(entry.get("result"))
            error_text = self._stringify(entry.get("error"))
            stdout_text = (entry.get("stdout") or "").strip()
            stderr_text = (entry.get("stderr") or "").strip()

            lines = [f"{name} • {status}", f"Started: {started}", f"Completed: {completed}"]
            if duration_line:
                lines.append(duration_line)
            if args_text:
                lines.append(f"Args: {args_text}")
            if result_text:
                lines.append(f"Result: {result_text}")
            if error_text:
                lines.append(f"Error: {error_text}")
            if stdout_text:
                lines.append("stdout:\n" + self._indent(stdout_text))
            if stderr_text:
                lines.append("stderr:\n" + self._indent(stderr_text))

            blocks.append("\n".join(lines))

        text = "\n\n".join(blocks).strip()
        return text or "No tool logs available."

    def _stringify(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            import json

            return json.dumps(value, ensure_ascii=False, indent=2)
        except Exception:
            return str(value)

    def _indent(self, value: str) -> str:
        return "\n".join(f"  {line}" if line else "" for line in value.splitlines())

    def _set_text(self, view: Gtk.TextView, text: str) -> None:
        buffer = view.get_buffer()
        if buffer is not None:
            buffer.set_text(text)
