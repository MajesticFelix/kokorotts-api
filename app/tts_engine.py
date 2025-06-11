import io
from kokoro import KPipeline
import soundfile as sf
from pydub import AudioSegment

pipeline = KPipeline(lang_code="a") # 'a' for American English

def synthesize(text: str, speaker: str, speed: float, format: str="wav") -> bytes:
    """Generate text-to-speech audio and return as bytes"""
    generator = pipeline(text, speaker, speed)

    for i, (gs, ps, audio) in enumerate(generator): # Note that gs represents the input string and ps represents the output tokens
        print(i, gs, ps)
        return audio_to_bytes(audio, format)
    
def audio_to_bytes(audio_data, target_format: str) -> bytes:
    """Convert targetted audio format array to bytes in memory (ffmpeg required for pydub)"""
    soundfile_formats = ["wav", "flac", "ogg"]
    pydub_formats = ["mp3", "opus", "pcm"] # ffmpeg required

    if target_format.lower() in soundfile_formats:
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, 24000, format=target_format.upper())
        buffer.seek(0)
        return buffer.getvalue()
    elif target_format.lower() in pydub_formats: 
        #Convert WAV to the targetted audios
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_data, 24000, format="WAV")
        wav_buffer.seek(0)

        audio = AudioSegment.from_wav(wav_buffer)
        output_buffer = io.BytesIO()
        
        if target_format.lower() == "mp3":
            audio.export(output_buffer, format="mp3", bitrate="192k")
        elif target_format.lower() == "opus":
            audio.export(output_buffer, format="opus", codec="libopus")
        elif target_format.lower() == "pcm": # Raw Audio Data
            audio.export(output_buffer, format="wav", parameters=["-acodec", "pcm_s16le"])
        
        output_buffer.seek(0)
        return output_buffer.getvalue()
    else:
        raise ValueError(f"Unsupported format: {target_format}")

# def synthesize_multi_voice_blend(text: str, voice_weights: dict, speed: float) -> str:
