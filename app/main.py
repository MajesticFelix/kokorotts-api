from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
from .tts_engine import synthesize
from huggingface_hub import list_repo_files

app = FastAPI()

class TTSRequest(BaseModel):
    text: str
    speaker: str
    speed: float

@app.post("/speak")
async def speak(req: TTSRequest):
    try:
        output_path = synthesize(req.text, req.speaker, req.speed)
        return FileResponse(output_path, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/voices")
async def get_voices():
    try:
        files = list_repo_files("hexgrad/Kokoro-82M")
        voice_files = [f for f in files if f.endswith(".pt") and "voices" in f]
        voices = [f.split("/")[-1].replace(".pt", "") for f in voice_files]
        return voices
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))