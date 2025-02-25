# modules/Speech_Services/base.py

from abc import ABC, abstractmethod

class BaseTTS(ABC):
    @abstractmethod
    async def text_to_speech(self, text: str):
        pass

    @abstractmethod
    def set_voice(self, voice: dict):
        pass

    @abstractmethod
    def get_voices(self) -> list:
        pass

    @abstractmethod
    def set_tts(self, value: bool):
        pass

    @abstractmethod
    def get_tts(self) -> bool:
        pass

class BaseSTT(ABC):
    @abstractmethod
    def listen(self):
        pass

    @abstractmethod
    def stop_listening(self):
        pass

    @abstractmethod
    def transcribe(self, audio_file: str) -> str:
        pass
