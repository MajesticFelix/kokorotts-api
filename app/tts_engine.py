import io
import torch
import numpy as np
from kokoro import KPipeline
import soundfile as sf
from pydub import AudioSegment
from typing import Dict, Iterator, List

pipeline = KPipeline(lang_code="a", device="cuda" if torch.cuda.is_available() else "cpu") # 'a' for American English

def chunk_text(text: str, initial_chunk_size: int = 1000) -> List[str]:
    """Split text into manageable chunks to avoid token limits"""
    if not text.strip():
        return []
    
    # Split text into sentences
    sentences = text.replace('\n', ' ').split('.')
    chunks = []
    current_chunk = []
    current_size = 0
    chunk_size = initial_chunk_size
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_length = len(sentence)
        
        # If sentence is too long, split it further
        if sentence_length > chunk_size:
            # Split long sentence into word-based pieces
            words = sentence.split()
            word_chunk = []
            word_chunk_size = 0
            
            for word in words:
                word_length = len(word) + 1  # +1 for space
                if word_chunk_size + word_length > chunk_size and word_chunk:
                    # Add current word chunk
                    chunks.append(' '.join(word_chunk) + '.')
                    word_chunk = [word]
                    word_chunk_size = word_length
                else:
                    word_chunk.append(word)
                    word_chunk_size += word_length
            
            # Add remaining words
            if word_chunk:
                chunks.append(' '.join(word_chunk) + '.')
        else:
            # Check if adding this sentence would exceed chunk size
            if current_size + sentence_length > chunk_size and current_chunk:
                # Finalize current chunk
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_size = sentence_length
            else:
                current_chunk.append(sentence)
                current_size += sentence_length
    
    # Add any remaining sentences
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
    
    return chunks

    
def audio_to_bytes(audio_data, target_format: str) -> bytes:
    """Convert targetted audio format array to bytes in memory (ffmpeg required for pydub)"""
    soundfile_formats = ["wav", "flac", "ogg"]
    pydub_formats = ["mp3", "opus"] # ffmpeg required

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
                "opus": {"format": "opus", "codec": "libopus"}
            }

            config = format_configs[target_format.lower()]
            audio.export(output_buffer, **config)
            
            output_buffer.seek(0)
            return output_buffer.getvalue()
    except Exception as e:
        raise ValueError(f"Audio conversion failed for {target_format}: {str(e)}")
    
    raise ValueError(f"Unsupported format: {target_format}")

def audio_to_raw_bytes(audio_data) -> bytes:
    """Convert audio data to raw bytes without any container format headers"""
    try:
        # Convert to 16-bit PCM format (standard for WAV)
        audio_int16 = (audio_data * 32767).astype(np.int16)
        return audio_int16.tobytes()
    except Exception as e:
        raise ValueError(f"Raw audio conversion failed: {str(e)}")

def create_wav_streaming_header(sample_rate: int = 24000, channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """Create a WAV header with large placeholder data size for streaming"""
    # WAV header structure with dummy large data size for streaming
    header = bytearray()
    
    # RIFF header
    header.extend(b'RIFF')
    header.extend((0x7FFFFFFF - 8).to_bytes(4, 'little'))  # Large placeholder for file size
    header.extend(b'WAVE')
    
    # fmt chunk
    header.extend(b'fmt ')
    header.extend((16).to_bytes(4, 'little'))  # fmt chunk size
    header.extend((1).to_bytes(2, 'little'))   # PCM format
    header.extend(channels.to_bytes(2, 'little'))  # Number of channels
    header.extend(sample_rate.to_bytes(4, 'little'))  # Sample rate
    header.extend((sample_rate * channels * bits_per_sample // 8).to_bytes(4, 'little'))  # Byte rate
    header.extend((channels * bits_per_sample // 8).to_bytes(2, 'little'))  # Block align
    header.extend(bits_per_sample.to_bytes(2, 'little'))  # Bits per sample
    
    # data chunk
    header.extend(b'data')
    header.extend((0x7FFFFFFF - 44).to_bytes(4, 'little'))  # Large placeholder for data size
    
    return bytes(header)

def synthesize(text: str, speaker: str, speed: float, format: str="wav") -> bytes:
    """Generate text-to-speech audio and return as bytes with automatic text chunking"""
    if not text.strip():
        raise ValueError("Text cannot be empty")
    
    try:
        # Automatically chunk text to avoid token limits
        text_chunks = chunk_text(text, initial_chunk_size=800)  # Conservative size for stability
        print(f"Split text into {len(text_chunks)} chunks for processing")
        
        all_audio_data = []
        
        for chunk_idx, chunk in enumerate(text_chunks):
            print(f"Processing chunk {chunk_idx + 1}/{len(text_chunks)}: {chunk[:50]}...")
            
            generator = pipeline(chunk, speaker, speed)
            for result in generator:
                audio = result.audio.cpu().numpy()  # Convert to numpy array
                print(f"Generated sub-chunk for chunk {chunk_idx + 1}: {len(audio)} samples")
                all_audio_data.append(audio)
        
        # Concatenate all audio data
        if len(all_audio_data) == 1:
            combined_audio = all_audio_data[0]
        else:
            combined_audio = np.concatenate(all_audio_data, axis=0)
        
        print(f"Combined {len(all_audio_data)} audio segments: {len(combined_audio)} total samples")
        return audio_to_bytes(combined_audio, format)
    except Exception as e:
        raise RuntimeError(f"Synthesis failed: {str(e)}")

def synthesize_with_voice_blend(text: str, voice_weights: Dict[str, float], speed: float, format: str = "wav") -> bytes:
    """Generate text-to-speech audio with blended voices and return as bytes with automatic text chunking"""
    if not text.strip():
        raise ValueError("Text cannot be empty")
    
    if not voice_weights:
        raise ValueError("Voice weights cannot be empty")
    
    # Normalize weights to sum to 1.0
    total_weight = sum(voice_weights.values())
    if total_weight <= 0:
        raise ValueError("Total voice weights must be positive")
    
    normalized_weights = {voice: weight / total_weight for voice, weight in voice_weights.items()}
    
    try:
        # Load and blend voice tensors
        blended_voice = None
        for voice_name, weight in normalized_weights.items():
            voice_tensor = pipeline.load_voice(voice_name)
            if blended_voice is None:
                blended_voice = weight * voice_tensor
            else:
                blended_voice += weight * voice_tensor
        
        # Automatically chunk text to avoid token limits
        text_chunks = chunk_text(text, initial_chunk_size=800)
        print(f"Voice blend: Split text into {len(text_chunks)} chunks for processing")
        
        all_audio_data = []
        
        for chunk_idx, chunk in enumerate(text_chunks):
            print(f"Processing voice blend chunk {chunk_idx + 1}/{len(text_chunks)}: {chunk[:50]}...")
            
            generator = pipeline(chunk, voice=blended_voice, speed=speed)
            for result in generator:
                audio = result.audio.cpu().numpy()  # Convert to numpy array
                print(f"Generated sub-chunk for blend chunk {chunk_idx + 1}: {len(audio)} samples")
                all_audio_data.append(audio)
        
        # Concatenate all audio data
        if len(all_audio_data) == 1:
            combined_audio = all_audio_data[0]
        else:
            combined_audio = np.concatenate(all_audio_data, axis=0)
        
        print(f"Combined {len(all_audio_data)} blended audio segments: {len(combined_audio)} total samples")
        return audio_to_bytes(combined_audio, format)
    except Exception as e:
        raise RuntimeError(f"Voice blend synthesis failed: {str(e)}")

def synthesize_streaming(text: str, speaker: str, speed: float, format: str = "wav") -> Iterator[bytes]:
    """Generate text-to-speech audio in streaming chunks with automatic text chunking"""
    if not text.strip():
        raise ValueError("Text cannot be empty")
    
    try:
        # Automatically chunk text to avoid token limits
        text_chunks = chunk_text(text, initial_chunk_size=800)
        print(f"Streaming: Split text into {len(text_chunks)} chunks for processing")
        
        chunk_counter = 0
        
        for chunk_idx, chunk in enumerate(text_chunks):
            print(f"Streaming chunk {chunk_idx + 1}/{len(text_chunks)}: {chunk[:50]}...")
            
            generator = pipeline(chunk, speaker, speed)
            for result in generator:
                chunk_counter += 1
                audio = result.audio.cpu().numpy()  # Convert to numpy array
                print(f"Streaming sub-chunk {chunk_counter}: {result.graphemes} -> {len(audio)} samples")
                
                if format.lower() == "wav":
                    if chunk_counter == 1:
                        # Send dummy WAV header first, then raw audio data
                        yield create_wav_streaming_header()
                        yield audio_to_raw_bytes(audio)
                    else:
                        # All subsequent chunks are raw audio data
                        yield audio_to_raw_bytes(audio)
                else:
                    # For non-WAV formats, each chunk needs complete file headers
                    yield audio_to_bytes(audio, format)
    except Exception as e:
        raise RuntimeError(f"Streaming synthesis failed: {str(e)}")

def synthesize_streaming_with_voice_blend(text: str, voice_weights: Dict[str, float], speed: float, format: str = "wav") -> Iterator[bytes]:
    """Generate text-to-speech audio with blended voices in streaming chunks with automatic text chunking"""
    if not text.strip():
        raise ValueError("Text cannot be empty")
    
    if not voice_weights:
        raise ValueError("Voice weights cannot be empty")
    
    # Normalize weights to sum to 1.0
    total_weight = sum(voice_weights.values())
    if total_weight <= 0:
        raise ValueError("Total voice weights must be positive")
    
    normalized_weights = {voice: weight / total_weight for voice, weight in voice_weights.items()}
    
    try:
        # Load and blend voice tensors
        blended_voice = None
        for voice_name, weight in normalized_weights.items():
            voice_tensor = pipeline.load_voice(voice_name)
            if blended_voice is None:
                blended_voice = weight * voice_tensor
            else:
                blended_voice += weight * voice_tensor
        
        # Automatically chunk text to avoid token limits
        text_chunks = chunk_text(text, initial_chunk_size=800)
        print(f"Streaming voice blend: Split text into {len(text_chunks)} chunks for processing")
        
        chunk_counter = 0
        
        for chunk_idx, chunk in enumerate(text_chunks):
            print(f"Streaming voice blend chunk {chunk_idx + 1}/{len(text_chunks)}: {chunk[:50]}...")
            
            generator = pipeline(chunk, voice=blended_voice, speed=speed)
            for result in generator:
                chunk_counter += 1
                audio = result.audio.cpu().numpy()  # Convert to numpy array
                print(f"Streaming blended sub-chunk {chunk_counter}: {result.graphemes} -> {len(audio)} samples")
                
                if format.lower() == "wav":
                    if chunk_counter == 1:
                        # Send dummy WAV header first, then raw audio data
                        yield create_wav_streaming_header()
                        yield audio_to_raw_bytes(audio)
                    else:
                        # All subsequent chunks are raw audio data
                        yield audio_to_raw_bytes(audio)
                else:
                    # For non-WAV formats, each chunk needs complete file headers
                    yield audio_to_bytes(audio, format)
    except Exception as e:
        raise RuntimeError(f"Voice blend streaming synthesis failed: {str(e)}")

