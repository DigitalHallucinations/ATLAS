"""Muse persona tool dispatch mapping."""

from __future__ import annotations

from modules.Tools.Base_Tools.time import get_current_info
from modules.Tools.Base_Tools.context_tracker import context_tracker
from modules.Tools.Base_Tools.story_weaver import StoryWeaver
from modules.Tools.Base_Tools.lyricist import Lyricist
from modules.Tools.Base_Tools.visual_prompt import VisualPrompt
from modules.Tools.Base_Tools.mood_map import MoodMap
from modules.Tools.Base_Tools.emotive_tagger import EmotiveTagger

_storyweaver = StoryWeaver()
_lyricist = Lyricist()
_visual_prompt = VisualPrompt()
_mood_map = MoodMap()
_emotive_tagger = EmotiveTagger()


function_map = {
    "get_current_info": get_current_info,
    "context_tracker": context_tracker,
    "storyweaver": _storyweaver.run,
    "lyricist": _lyricist.run,
    "visual_prompt": _visual_prompt.run,
    "mood_map": _mood_map.run,
    "emotive_tagger": _emotive_tagger.run,
}


__all__ = ["function_map"]
