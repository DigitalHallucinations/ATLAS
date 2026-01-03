import json
from types import SimpleNamespace

from modules.Speech_Services.speech_manager import SpeechManager
from modules.Speech_Services.whisper_stt import WhisperSTT


class FakeWhisper(WhisperSTT):
    def __init__(self):  # pragma: no cover - prevent base initialization
        pass

    def transcribe(self, audio_file=None, *, export_options=None, **_kwargs):
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello"},
            {"start": 1.0, "end": 2.0, "text": "world"},
        ]
        text = "Hello world"
        self.export_transcript(text, segments, export_options)
        return text


def test_whisper_transcription_exports(tmp_path):
    manager = SpeechManager(SimpleNamespace(config={}))
    fake_whisper = FakeWhisper()
    manager.stt_services["whisper_local"] = fake_whisper

    audio_path = tmp_path / "sample.wav"
    audio_path.write_text("dummy")

    export_options = {
        "formats": ["txt", "json", "srt"],
        "destination": str(tmp_path),
        "metadata": {"session": "unit"},
    }

    transcript = manager.transcribe(
        str(audio_path),
        provider="whisper_local",
        export_options=export_options,
    )

    assert transcript == "Hello world"

    txt_path = tmp_path / "sample.txt"
    json_path = tmp_path / "sample.json"
    srt_path = tmp_path / "sample.srt"

    assert txt_path.exists()
    assert json_path.exists()
    assert srt_path.exists()

    txt_content = txt_path.read_text(encoding="utf-8")
    assert "# Metadata:" in txt_content
    assert "# provider: whisper_local" in txt_content
    assert "# session: unit" in txt_content
    assert "Hello world" in txt_content

    json_content = json_path.read_text(encoding="utf-8")
    data = json.loads(json_content)
    assert data["transcript"]["text"] == "Hello world"
    assert data["transcript"]["segments"][0]["text"] == "Hello"
    metadata = data["metadata"]
    assert metadata["provider"] == "whisper_local"
    assert metadata["audio_file"] == str(audio_path)
    assert metadata["session"] == "unit"
    assert metadata["timestamp"]

    srt_content = srt_path.read_text(encoding="utf-8")
    assert "Hello" in srt_content
    assert "00:00:00,000" in srt_content
