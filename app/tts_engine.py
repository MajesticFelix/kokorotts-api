import os
import uuid
from kokoro import KPipeline
import soundfile as sf
from pydub import AudioSegment

pipeline = KPipeline(lang_code="a") # 'a' for American English

def synthesize(text: str, speaker: str, speed: float, format: str="wav") -> str:
    """Generate text-to-speech audio"""
    file_name = f"{uuid.uuid4().hex}"
    wav_path = os.path.join("output", f"{file_name}.wav")
    generator = pipeline(text, speaker, speed)

    for i, (gs, ps, audio) in enumerate(generator): # Note that gs represents the input string and ps represents the output tokens
        print(i, gs, ps)
        sf.write(wav_path, audio, 24000)
        break
    
    if format.lower() == "wav":
        return wav_path
    else:
        return convert_audio_format(wav_path, format, file_name)
    
def convert_audio_format(wav_path: str, target_format: str, base_filename: str) -> str:
    """Convert WAV format to targetted format using pydub (ffmpeg required)"""
    output_path = os.path.join("output", f"{base_filename}.{target_format}")
    audio = AudioSegment.from_wav(wav_path)

    if target_format.lower() == "mp3":
        audio.export(output_path, format="mp3", bitrate="192k")
    elif target_format.lower() == "flac":
        audio.export(output_path, format="flac")
    elif target_format.lower() == "opus":
        audio.export(output_path, format="opus", codec="libopus")
    elif target_format.lower() == "pcm": # Raw Audio Data
        audio.export(output_path, format="wav", parameters=["-acodec", "pcm_s16le"])
    
    # Clean up temp WAV file if it was converted
    if target_format.lower() != "wav":
        os.remove(wav_path)
    
    return output_path

# def synthesize_multi_voice_blend(text: str, voice_weights: dict, speed: float) -> str:
