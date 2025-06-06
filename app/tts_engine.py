import os
import uuid
from kokoro import KPipeline
import soundfile as sf

pipeline = KPipeline(lang_code="a") # 'a' for American English

def synthesize(text: str, speaker: str, speed: float) -> str:
    file_name = f"{uuid.uuid4().hex}.wav"
    output_path = os.path.join("output", file_name)
    generator = pipeline(text, speaker, speed)
    for i, (gs, ps, audio) in enumerate(generator):
        print(i, gs, ps)
        sf.write(output_path, audio, 24000)
    return output_path