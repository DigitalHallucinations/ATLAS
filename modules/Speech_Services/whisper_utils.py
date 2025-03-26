"""
Module: whisper_utils.py
Description:
    Provides utility functions for processing and exporting Whisper STT transcription results.
    Supports exporting to TXT, JSON, and SRT with optional metadata.
    
Author: Jeremy Shows - Digital Hallucinations
Date: 05-11-2025
"""
import json

def export_to_txt(transcript, filename, metadata=None):
    """
    Exports transcript (plain text) to a .txt file.
    Optionally includes metadata at the top.
    """
    with open(filename, "w", encoding="utf-8") as f:
        if metadata:
            f.write("# Metadata:\n")
            for key, value in metadata.items():
                f.write(f"# {key}: {value}\n")
            f.write("\n")
        if isinstance(transcript, list):
            text = "\n".join(segment.get("text", "") for segment in transcript)
        else:
            text = transcript
        f.write(text)

def export_to_json(transcript, filename, metadata=None):
    """
    Exports transcript and metadata to a .json file.
    """
    output = {"transcript": transcript}
    if metadata:
        output["metadata"] = metadata
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

def export_to_srt(transcript, filename):
    """
    Exports transcript segments to SRT subtitle format.
    Expects transcript to be a list of segments with "start", "end", and "text" keys.
    """
    def format_timestamp(seconds):
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = seconds % 60
        millis = int((secs - int(secs)) * 1000)
        return f"{hrs:02}:{mins:02}:{int(secs):02},{millis:03}"
    
    with open(filename, "w", encoding="utf-8") as f:
        for i, segment in enumerate(transcript, start=1):
            start = format_timestamp(segment.get("start", 0))
            end = format_timestamp(segment.get("end", 0))
            text = segment.get("text", "")
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
