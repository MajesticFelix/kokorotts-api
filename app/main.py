from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
from .tts_engine import synthesize

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