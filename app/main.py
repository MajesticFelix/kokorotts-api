from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
from .tts_engine import synthesize
from huggingface_hub import list_repo_files
from enum import Enum

app = FastAPI(title="Kokoro TTS API", version="1.0.0")

class AudioFormat(str, Enum):
    wav = "wav"
    mp3 = "mp3"
    flac = "flac"
    opus = "opus"
    pcm = "pcm"

class TTSRequest(BaseModel):
    text: str
    speaker: str = "af_heart"
    speed: float = 1.0
    format: AudioFormat = AudioFormat.wav

@app.get("/")
async def root():
    return {"message": "Kokoro TTS API is running!", "docs": "/docs"}

@app.post("/speak")
async def speak(req: TTSRequest):
    try:
        output_path = synthesize(req.text, req.speaker, req.speed, req.format.value)
        media_types = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg", 
            "flac": "audio/flac",
            "opus": "audio/opus",
            "pcm": "audio/pcm"
        }
        media_type = media_types.get(req.format.value, "audio/wav")
        file_name = f"speech_{req.speaker}.{req.format.value}"
        headers = {"content-disposition": f'attachment; filename="{file_name}"'}
        return FileResponse(output_path, media_type=media_type, headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/voices")
async def get_voices():
    try:
        files = list_repo_files(repo_id="hexgrad/Kokoro-82M")
        voice_files = [f for f in files if f.endswith(".pt") and "voices" in f]
        voices = [f.split("/")[-1].replace(".pt", "") for f in voice_files]
        return voices
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))