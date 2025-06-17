from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from .tts_engine import synthesize, synthesize_with_voice_blend, synthesize_streaming, synthesize_streaming_with_voice_blend
from typing import Dict
from huggingface_hub import list_repo_files
from enum import Enum
import io

app = FastAPI(title="Kokoro TTS API", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AudioFormat(str, Enum):
    wav = "wav"
    flac = "flac"
    ogg = "ogg"
    mp3 = "mp3"
    opus = "opus"
    pcm = "pcm"

class TTSRequest(BaseModel):
    text: str
    speaker: str = "af_heart"
    speed: float = 1.0
    format: AudioFormat = AudioFormat.wav

class TTSBlendRequest(BaseModel):
    text: str
    voice_weights: Dict[str, float]
    speed: float = 1.0
    format: AudioFormat = AudioFormat.wav

class TTSStreamRequest(BaseModel):
    text: str
    speaker: str = "af_heart"
    speed: float = 1.0
    format: AudioFormat = AudioFormat.wav
    split_pattern: str = r'[.!?]+\s*'

class TTSStreamBlendRequest(BaseModel):
    text: str
    voice_weights: Dict[str, float]
    speed: float = 1.0
    format: AudioFormat = AudioFormat.wav
    split_pattern: str = r'[.!?]+\s*'

@app.get("/")
async def root():
    return {"message": "Kokoro TTS API is running!", "docs": "/docs", "test_streaming": "/test"}

@app.get("/test", response_class=HTMLResponse)
async def test_page():
    with open("static/test_streaming.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/speak")
async def speak(req: TTSRequest):
    try:
        audio_bytes = synthesize(req.text, req.speaker, req.speed, req.format.value)
        audio_stream = io.BytesIO(audio_bytes)

        media_types = {
            "wav": "audio/wav",
            "flac": "audio/flac",
            "ogg": "audio/ogg",
            "mp3": "audio/mpeg", 
            "opus": "audio/opus",
            "pcm": "audio/pcm"
        }
        media_type = media_types.get(req.format.value, "audio/wav")
        file_name = f"speech_{req.speaker}.{req.format.value}"
        headers = {"Content-Disposition": f'attachment; filename="{file_name}"'}
        return StreamingResponse(audio_stream, media_type=media_type, headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/speak-blend")
async def speak_blend(req: TTSBlendRequest):
    try:
        audio_bytes = synthesize_with_voice_blend(req.text, req.voice_weights, req.speed, req.format.value)
        audio_stream = io.BytesIO(audio_bytes)

        media_types = {
            "wav": "audio/wav",
            "flac": "audio/flac",
            "ogg": "audio/ogg",
            "mp3": "audio/mpeg", 
            "opus": "audio/opus",
            "pcm": "audio/pcm"
        }
        media_type = media_types.get(req.format.value, "audio/wav")
        voice_names = "_".join(req.voice_weights.keys())
        file_name = f"speech_blend_{voice_names}.{req.format.value}"
        headers = {"Content-Disposition": f'attachment; filename="{file_name}"'}
        return StreamingResponse(audio_stream, media_type=media_type, headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/speak-stream")
async def speak_stream(req: TTSStreamRequest):
    try:
        def generate_audio():
            for audio_chunk in synthesize_streaming(req.text, req.speaker, req.speed, req.format.value, req.split_pattern):
                yield audio_chunk
        
        media_types = {
            "wav": "audio/wav",
            "flac": "audio/flac",
            "ogg": "audio/ogg",
            "mp3": "audio/mpeg", 
            "opus": "audio/opus",
            "pcm": "audio/pcm"
        }
        media_type = media_types.get(req.format.value, "audio/wav")
        # No Content-Disposition header for true streaming
        return StreamingResponse(generate_audio(), media_type=media_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/speak-stream-blend")
async def speak_stream_blend(req: TTSStreamBlendRequest):
    try:
        def generate_audio():
            for audio_chunk in synthesize_streaming_with_voice_blend(req.text, req.voice_weights, req.speed, req.format.value, req.split_pattern):
                yield audio_chunk
        
        media_types = {
            "wav": "audio/wav",
            "flac": "audio/flac",
            "ogg": "audio/ogg",
            "mp3": "audio/mpeg", 
            "opus": "audio/opus",
            "pcm": "audio/pcm"
        }
        media_type = media_types.get(req.format.value, "audio/wav")
        # No Content-Disposition header for true streaming
        return StreamingResponse(generate_audio(), media_type=media_type)
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