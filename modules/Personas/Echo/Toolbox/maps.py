"""Echo persona tool dispatch mapping."""

from __future__ import annotations

from modules.Tools.Base_Tools.time import get_current_info
from modules.Tools.Base_Tools.context_tracker import context_tracker
from modules.Tools.Base_Tools.tone_analyzer import ToneAnalyzer
from modules.Tools.Base_Tools.reflective_prompt import ReflectivePrompt
from modules.Tools.Base_Tools.memory_recall import MemoryRecall
from modules.Tools.Base_Tools.conflict_resolver import ConflictResolver

_tone_analyzer = ToneAnalyzer()
_reflective_prompt = ReflectivePrompt()
_memory_recall = MemoryRecall()
_conflict_resolver = ConflictResolver()


function_map = {
    "get_current_info": get_current_info,
    "context_tracker": context_tracker,
    "tone_analyzer": _tone_analyzer.run,
    "reflective_prompt": _reflective_prompt.run,
    "memory_recall": _memory_recall.run,
    "conflict_resolver": _conflict_resolver.run,
}


__all__ = ["function_map"]
