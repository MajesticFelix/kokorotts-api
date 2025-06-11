import io
import torch
from kokoro import KPipeline
import soundfile as sf
from pydub import AudioSegment

pipeline = KPipeline(lang_code="a", device="cuda" if torch.cuda.is_available() else "cpu") # 'a' for American English

def synthesize(text: str, speaker: str, speed: float, format: str="wav") -> bytes:
    """Generate text-to-speech audio and return as bytes"""
    if not text.strip():
        raise ValueError("Text cannot be empty")
    
    try:
        generator = pipeline(text, speaker, speed)
        for i, (gs, ps, audio) in enumerate(generator): # Note that gs represents the input string and ps represents the output tokens
            print(f"Generated chunk {i}: {gs} -> {len(audio)} samples")
            return audio_to_bytes(audio, format)
    except Exception as e:
        raise RuntimeError(f"Synthesis failed: {str(e)}")
    
def audio_to_bytes(audio_data, target_format: str) -> bytes:
    """Convert targetted audio format array to bytes in memory (ffmpeg required for pydub)"""
    soundfile_formats = ["wav", "flac", "ogg"]
    pydub_formats = ["mp3", "opus", "pcm"] # ffmpeg required

    try: 
        if target_format.lower() in soundfile_formats:
            buffer = io.BytesIO()
            sf.write(buffer, audio_data, 24000, format=target_format.upper())
            buffer.seek(0)
            return buffer.getvalue()
        elif target_format.lower() in pydub_formats: 
            # Convert WAV to the targetted audios
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio_data, 24000, format="WAV")
            wav_buffer.seek(0)

            audio = AudioSegment.from_wav(wav_buffer)
            output_buffer = io.BytesIO()
            
            format_configs = {
                "mp3": {"format": "mp3", "bitrate": "192k"},
                "opus": {"format": "opus", "codec": "libopus"},
                "pcm": {"format": "wav", "parameters": ["-acodec", "pcm_s16le"]}
            }

            config = format_configs[target_format.lower()]
            audio.export(output_buffer, **config)
            
            output_buffer.seek(0)
            return output_buffer.getvalue()
    except Exception as e:
        raise ValueError(f"Audio conversion failed for {target_format}: {str(e)}")
    
    raise ValueError(f"Unsupported format: {target_format}")

# def synthesize_multi_voice_blend(text: str, voice_weights: dict, speed: float) -> str:
