# modules\Speech_Services\Ggl_stt.py

import os
import sounddevice as sd
import numpy as np
import soundfile as sf
from google.cloud import speech_v1p1beta1 as speech
from modules.logging.logger import setup_logger

logger = setup_logger('Ggl_stt.py')

class GoogleSTT:
    def __init__(
        self,
        fs=16000,
        sample_rate_hertz=None,
        enable_automatic_punctuation=True,
    ):
        logger.debug("Initializing GoogleSTT")
        effective_sample_rate = sample_rate_hertz if sample_rate_hertz is not None else fs
        if effective_sample_rate is None:
            effective_sample_rate = 16000
        self.client = speech.SpeechClient()
        self.config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=effective_sample_rate,
            language_code="en-US",
            enable_automatic_punctuation=enable_automatic_punctuation
        )

        self.fs = effective_sample_rate
        self.frames = []
        self.recording = False
        self.stream = sd.InputStream(callback=self.callback, channels=1, samplerate=self.fs)

    def callback(self, indata, frames, time, status):
        if self.recording:
            self.frames.append(indata.copy())

    def listen(self):  
        logger.debug("Listening...")
        self.frames = []
        self.recording = True
        self.stream = sd.InputStream(callback=self.callback, channels=1, samplerate=self.fs)
        self.stream.start()

    def stop_listening(self):
        logger.debug("Stopping listening")
        if self.recording:
            self.stream.stop()
            self.recording = False
            self.save_recording()
            logger.debug("Recording stopped")
        else:
            logger.warning("Tried to stop listening, but was not recording")
        self.stream.close()
        self.frames = []

    def save_recording(self, filename='output.wav'):
        logger.debug("Saving recording")
        output_dir = 'data/audio/stt_output'
        os.makedirs(output_dir, exist_ok=True)  
        output_file = os.path.join(output_dir, filename)

        if self.frames:
            data = np.concatenate(self.frames)
            sf.write(output_file, data, self.fs)
            logger.debug("Audio recorded and saved as %s", output_file)
        else:
            logger.warning("No frames to save")

    def transcribe(self, audio_file):
        logger.debug("Transcribing file %s", audio_file)
        audio_file_path = os.path.join('data/audio/stt_output', audio_file)

        if not os.path.exists(audio_file_path):
            logger.error(f"Audio file not found: {audio_file_path}")
            return "No audio file to transcribe"

        try:
            with open(audio_file_path, 'rb') as audio:
                audio_content = audio.read()
        except FileNotFoundError:
            logger.error(f"Audio file not found: {audio_file_path}")
            return "No audio file to transcribe"

        audio = speech.RecognitionAudio(content=audio_content)
        response = self.client.recognize(config=self.config, audio=audio)

        transcript = []
        for result in response.results:
            transcript.append(result.alternatives[0].transcript)

        # Delete the audio file after transcription
        os.remove(audio_file_path)
        logger.debug("Deleted audio file %s", audio_file_path)

        return ' '.join(transcript)