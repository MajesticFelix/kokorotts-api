"""Text-to-speech engine using KokoroTTS model.

This module provides high-level functions for text-to-speech synthesis
with support for multiple voices, languages, streaming, and captions.
"""

import io
import logging
import gc
import re
from typing import Dict, Iterator, List, NamedTuple, Union, Optional

import numpy as np
import soundfile as sf
import torch
from kokoro import KPipeline
from pydub import AudioSegment

# Configure logging
logger = logging.getLogger(__name__)

# Global pipeline cache for different languages
_pipelines: Dict[str, KPipeline] = {}
_device = "cuda" if torch.cuda.is_available() else "cpu"

# Configuration Constants
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_CHUNK_SIZE = 800
MAX_CHUNK_SIZE = 1000
STREAMING_CHUNK_SIZE = 800
AUDIO_SAMPLE_RATE = 24000

# Audio format configurations
AUDIO_FORMAT_CONFIG = {
    "soundfile_formats": ["wav", "flac", "ogg"],
    "pydub_formats": ["mp3", "opus"],
    "format_configs": {
        "mp3": {"format": "mp3", "bitrate": "192k"},
        "opus": {"format": "opus", "codec": "libopus"}
    }
}

# Data structures for timing information
class WordTiming(NamedTuple):
    """Word-level timing information for TTS output.
    
    Attributes:
        text: The word text
        start_time: Start time in seconds
        end_time: End time in seconds
        phonemes: Phonetic representation (IPA or graphemes)
    """
    text: str
    start_time: float
    end_time: float
    phonemes: str

class SynthesisResult(NamedTuple):
    """Complete synthesis result with audio and timing data.
    
    Attributes:
        word_timings: List of word-level timing information
        audio_bytes: Encoded audio data
        total_duration: Total audio duration in seconds
        sample_rate: Audio sample rate in Hz
    """
    word_timings: List[WordTiming]
    audio_bytes: bytes
    total_duration: float
    sample_rate: int

class StreamingCaptionChunk(NamedTuple):
    """Streaming chunk with caption data.
    
    Attributes:
        word_timings: Word timings for this chunk
        audio_data: Encoded audio data for this chunk
        chunk_number: Sequential chunk number
        is_final: Whether this is the final chunk
    """
    word_timings: List[WordTiming]
    audio_data: bytes
    chunk_number: int
    is_final: bool

def get_pipeline(lang_code: str = "a") -> KPipeline:
    """Get or create a pipeline for the specified language code.
    
    Args:
        lang_code: Language code (e.g., 'a' for American English)
        
    Returns:
        KPipeline instance for the specified language
        
    Raises:
        RuntimeError: If pipeline initialization fails
    """
    if lang_code not in _pipelines:
        try:
            logger.info(f"Initializing pipeline for language: {lang_code}")
            _pipelines[lang_code] = KPipeline(lang_code=lang_code, device=_device)
        except Exception as e:
            logger.error(f"Failed to initialize pipeline for {lang_code}: {e}")
            raise RuntimeError(f"Pipeline initialization failed: {e}") from e
    return _pipelines[lang_code]

# Initialize default pipeline for backward compatibility
pipeline = get_pipeline("a")

def text_segmentation(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[str]:
    """Intelligently segment text for optimal TTS processing.
    
    Uses sentence boundaries, paragraph breaks, and semantic markers
    to create natural-sounding chunks that avoid cutting mid-sentence.
    
    Args:
        text: Input text to segment
        chunk_size: Target size for each chunk
        
    Returns:
        List of text segments optimized for TTS
        
    Raises:
        ValueError: If text is empty
    """
    if not text or not text.strip():
        logger.warning("Empty text provided to text_segmentation")
        return []
    
    # Normalize whitespace and line breaks
    normalized_text = re.sub(r'\s+', ' ', text.strip())
    
    # Split by strong paragraph boundaries first
    paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', normalized_text)
    
    chunks = []
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
            
        # If paragraph is short enough, use as-is
        if len(paragraph) <= chunk_size:
            chunks.append(paragraph.strip())
            continue
        
        # Split long paragraphs by sentence boundaries
        # Enhanced sentence splitting that handles abbreviations
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, paragraph)
        
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_length = len(sentence)
            
            # If single sentence is too long, split by clauses/commas
            if sentence_length > chunk_size:
                clause_chunks = _split_long_sentence(sentence, chunk_size)
                chunks.extend(clause_chunks)
                continue
            
            # Check if adding sentence would exceed chunk size
            if current_size + sentence_length + 1 > chunk_size and current_chunk:
                # Finalize current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                current_chunk = [sentence]
                current_size = sentence_length
            else:
                current_chunk.append(sentence)
                current_size += sentence_length + 1  # +1 for space
        
        # Add any remaining sentences
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
    
    return chunks

def _split_long_sentence(sentence: str, max_size: int) -> List[str]:
    """Split overly long sentences by clauses and phrases.
    
    Args:
        sentence: Long sentence to split
        max_size: Maximum size for each piece
        
    Returns:
        List of sentence fragments
    """
    # Try splitting by natural pause points
    clause_patterns = [
        r',\s+',  # Commas
        r';\s+',  # Semicolons
        r'\s+(?:and|or|but|however|therefore|moreover)\s+',  # Conjunctions
        r'\s+(?:when|where|while|because|since|although)\s+',  # Subordinating conjunctions
    ]
    
    parts = [sentence]
    
    for pattern in clause_patterns:
        new_parts = []
        for part in parts:
            if len(part) <= max_size:
                new_parts.append(part)
            else:
                # Split by current pattern
                split_parts = re.split(f'({pattern})', part)
                current_piece = ""
                
                for piece in split_parts:
                    if len(current_piece + piece) <= max_size:
                        current_piece += piece
                    else:
                        if current_piece:
                            new_parts.append(current_piece.strip())
                        current_piece = piece
                
                if current_piece:
                    new_parts.append(current_piece.strip())
        
        parts = [p for p in new_parts if p.strip()]
        
        # If all parts are now small enough, we're done
        if all(len(p) <= max_size for p in parts):
            break
    
    # Final fallback: split by words if still too long
    final_parts = []
    for part in parts:
        if len(part) <= max_size:
            final_parts.append(part)
        else:
            words = part.split()
            current_chunk = []
            current_size = 0
            
            for word in words:
                word_length = len(word) + 1  # +1 for space
                if current_size + word_length > max_size and current_chunk:
                    final_parts.append(' '.join(current_chunk))
                    current_chunk = [word]
                    current_size = word_length
                else:
                    current_chunk.append(word)
                    current_size += word_length
            
            if current_chunk:
                final_parts.append(' '.join(current_chunk))
    
    return [p.strip() for p in final_parts if p.strip()]

# Backward compatibility
def chunk_text(text: str, initial_chunk_size: int = MAX_CHUNK_SIZE) -> List[str]:
    """Legacy function for backward compatibility."""
    return text_segmentation(text, initial_chunk_size)

    
def audio_to_bytes(audio_data: np.ndarray, target_format: str) -> bytes:
    """Convert audio data array to bytes in specified format.
    
    Args:
        audio_data: NumPy array containing audio samples
        target_format: Target audio format (wav, mp3, flac, ogg, opus)
        
    Returns:
        Encoded audio bytes
        
    Raises:
        ValueError: If format is unsupported or conversion fails
    """
    target_format_lower = target_format.lower()
    
    try:
        if target_format_lower in AUDIO_FORMAT_CONFIG["soundfile_formats"]:
            buffer = io.BytesIO()
            sf.write(buffer, audio_data, DEFAULT_SAMPLE_RATE, format=target_format.upper())
            buffer.seek(0)
            return buffer.getvalue()
        elif target_format_lower in AUDIO_FORMAT_CONFIG["pydub_formats"]:
            # Convert WAV to the target format
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio_data, DEFAULT_SAMPLE_RATE, format="WAV")
            wav_buffer.seek(0)

            audio = AudioSegment.from_wav(wav_buffer)
            output_buffer = io.BytesIO()
            
            config = AUDIO_FORMAT_CONFIG["format_configs"][target_format_lower]
            audio.export(output_buffer, **config)
            
            output_buffer.seek(0)
            return output_buffer.getvalue()
    except Exception as e:
        logger.error(f"Audio conversion failed for {target_format}: {e}")
        raise ValueError(f"Audio conversion failed for {target_format}: {e}") from e
    
    raise ValueError(f"Unsupported format: {target_format}")

def audio_to_raw_bytes(audio_data: np.ndarray) -> bytes:
    """Convert audio data to raw bytes without any container format headers.
    
    Args:
        audio_data: NumPy array containing audio samples
        
    Returns:
        Raw audio bytes in 16-bit PCM format
        
    Raises:
        ValueError: If conversion fails
    """
    try:
        # Convert to 16-bit PCM format (standard for WAV)
        audio_int16 = (audio_data * 32767).astype(np.int16)
        return audio_int16.tobytes()
    except Exception as e:
        logger.error(f"Raw audio conversion failed: {e}")
        raise ValueError(f"Raw audio conversion failed: {e}") from e

def create_wav_streaming_header(
    sample_rate: int = DEFAULT_SAMPLE_RATE, 
    channels: int = 1, 
    bits_per_sample: int = 16
) -> bytes:
    """Create a WAV header with large placeholder data size for streaming.
    
    Args:
        sample_rate: Audio sample rate in Hz
        channels: Number of audio channels
        bits_per_sample: Bits per audio sample
        
    Returns:
        WAV header bytes
    """
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

def extract_word_timings(result, chunk_offset: float = 0.0) -> List[WordTiming]:
    """Extract per-word timing information from KokoroTTS result object.
    
    Args:
        result: KokoroTTS synthesis result object
        chunk_offset: Time offset to add to all timestamps
        
    Returns:
        List of WordTiming objects with text and timing information
    """
    word_timings = []
    
    try:
        if hasattr(result, 'tokens') and result.tokens:
            for token in result.tokens:
                if hasattr(token, 'text') and hasattr(token, 'start_ts') and hasattr(token, 'end_ts'):
                    # Skip empty or whitespace-only tokens
                    if token.text.strip():
                        phonemes = getattr(token, 'phonemes', '') or getattr(token, 'graphemes', '') or ''
                        word_timing = WordTiming(
                            text=token.text,
                            start_time=token.start_ts + chunk_offset,
                            end_time=token.end_ts + chunk_offset,
                            phonemes=phonemes
                        )
                        word_timings.append(word_timing)
        else:
            logger.warning("No token timing information available in result object")
    except Exception as e:
        logger.error(f"Error extracting word timings: {e}")
    
    return word_timings

def calculate_audio_duration(audio_data: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> float:
    """Calculate duration of audio data in seconds.
    
    Args:
        audio_data: NumPy array containing audio samples
        sample_rate: Audio sample rate in Hz
        
    Returns:
        Duration in seconds
    """
    return len(audio_data) / sample_rate

def should_use_batch_processing(text: str, memory_limit_mb: float = 1024) -> bool:
    """Determine if text should be processed in batches.
    
    Args:
        text: Input text to analyze
        memory_limit_mb: Memory limit in MB
        
    Returns:
        True if batch processing is recommended
    """
    # Simple heuristic: if text is very long, use batch processing
    text_length = len(text)
    # Rough estimate: ~2.5MB per 1000 characters with captions
    estimated_mb = (text_length / 1000) * 2.5 * 1.5
    return estimated_mb > memory_limit_mb

def _batch_synthesize(
    text_chunks: List[str],
    voice_spec: Union[str, Dict[str, float]],
    prepared_voice,
    speed: float,
    format: str,
    lang_code: str,
    include_captions: bool,
    lang_pipeline: KPipeline
) -> Union[bytes, SynthesisResult]:
    """Process very long texts in memory-efficient batches.
    
    Args:
        text_chunks: Pre-segmented text chunks
        voice_spec: Voice specification
        prepared_voice: Prepared voice tensor or name
        speed: Speech speed
        format: Audio format
        lang_code: Language code
        include_captions: Whether to include captions
        lang_pipeline: KPipeline instance
        
    Returns:
        Synthesized audio or SynthesisResult
    """
    voice_type = "blended" if isinstance(voice_spec, dict) else "single"
    logger.info(f"Using batch processing for {len(text_chunks)} chunks ({voice_type} voice)")
    
    # Process in smaller batches to manage memory
    batch_size = 5  # Process 5 chunks at a time
    all_audio_segments = []
    all_word_timings = [] if include_captions else None
    chunk_time_offset = 0.0 if include_captions else 0.0
    
    for batch_start in range(0, len(text_chunks), batch_size):
        batch_end = min(batch_start + batch_size, len(text_chunks))
        batch_chunks = text_chunks[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_start//batch_size + 1}/{(len(text_chunks) + batch_size - 1)//batch_size}")
        
        batch_audio_data = []
        batch_word_timings = [] if include_captions else None
        
        for chunk_idx, chunk in enumerate(batch_chunks):
            abs_chunk_idx = batch_start + chunk_idx
            logger.debug(f"Processing chunk {abs_chunk_idx + 1}/{len(text_chunks)}: {chunk[:50]}...")
            
            if include_captions:
                chunk_audio_data = []
                chunk_word_timings = []
            
            # Use prepared voice (string for single voice, tensor for blended)
            if isinstance(prepared_voice, str):
                generator = lang_pipeline(chunk, prepared_voice, speed)
            else:
                generator = lang_pipeline(chunk, voice=prepared_voice, speed=speed)
                
            for result in generator:
                audio = result.audio.cpu().numpy()
                
                if include_captions:
                    # Extract timing information
                    sub_chunk_timings = extract_word_timings(result, chunk_time_offset)
                    chunk_word_timings.extend(sub_chunk_timings)
                    chunk_audio_data.append(audio)
                    
                    # Update time offset for next sub-chunk
                    chunk_time_offset += calculate_audio_duration(audio)
                else:
                    batch_audio_data.append(audio)
            
            # Handle audio data for captions
            if include_captions and chunk_audio_data:
                chunk_combined_audio = np.concatenate(chunk_audio_data) if len(chunk_audio_data) > 1 else chunk_audio_data[0]
                batch_audio_data.append(chunk_combined_audio)
                batch_word_timings.extend(chunk_word_timings)
        
        # Combine batch audio and clear memory
        if batch_audio_data:
            batch_combined = np.concatenate(batch_audio_data) if len(batch_audio_data) > 1 else batch_audio_data[0]
            
            # Convert to bytes immediately to free memory
            batch_audio_bytes = audio_to_bytes(batch_combined, format)
            all_audio_segments.append(batch_audio_bytes)
            
            if include_captions and batch_word_timings:
                all_word_timings.extend(batch_word_timings)
            
            # Force garbage collection to free memory
            del batch_audio_data, batch_combined
            if include_captions:
                del chunk_audio_data
            gc.collect()
            
            logger.debug(f"Batch {batch_start//batch_size + 1} completed, memory freed")
    
    # Combine all audio segments
    logger.info(f"Combining {len(all_audio_segments)} audio segments")
    
    if not all_audio_segments:
        raise RuntimeError("No audio data generated")
    
    if include_captions:
        # For captions, we need to reconstruct the final audio
        # This is memory-intensive but necessary for accurate timing
        logger.warning("Combining audio segments for captioned response - high memory usage")
        
        # Load all segments back to numpy arrays
        audio_arrays = []
        for segment_bytes in all_audio_segments:
            # Convert back to numpy for concatenation
            segment_buffer = io.BytesIO(segment_bytes)
            if format.lower() == 'wav':
                segment_array, _ = sf.read(segment_buffer)
            else:
                # For other formats, convert via pydub
                from pydub import AudioSegment as AS
                segment_audio = AS.from_file(segment_buffer, format=format)
                segment_array = np.array(segment_audio.get_array_of_samples()).astype(np.float32) / 32768.0
            audio_arrays.append(segment_array)
        
        combined_audio = np.concatenate(audio_arrays)
        total_duration = calculate_audio_duration(combined_audio)
        final_audio_bytes = audio_to_bytes(combined_audio, format)
        
        # Clean up
        del audio_arrays, combined_audio
        gc.collect()
        
        logger.info(f"Generated {len(all_word_timings)} words with timing information for {voice_type} voice")
        
        return SynthesisResult(
            word_timings=all_word_timings,
            audio_bytes=final_audio_bytes,
            total_duration=total_duration,
            sample_rate=AUDIO_SAMPLE_RATE
        )
    else:
        # For audio-only, concatenate the byte segments
        if len(all_audio_segments) == 1:
            return all_audio_segments[0]
        
        # Combine multiple audio byte segments
        # This is format-dependent
        if format.lower() == 'wav':
            # For WAV, we can efficiently combine
            return _combine_wav_segments(all_audio_segments)
        else:
            # For other formats, load and re-encode
            return _combine_audio_segments(all_audio_segments, format)

def _combine_wav_segments(wav_segments: List[bytes]) -> bytes:
    """Efficiently combine WAV audio segments.
    
    Args:
        wav_segments: List of WAV audio byte segments
        
    Returns:
        Combined WAV bytes
    """
    if len(wav_segments) == 1:
        return wav_segments[0]
    
    # Load all segments and combine
    audio_arrays = []
    for segment_bytes in wav_segments:
        segment_buffer = io.BytesIO(segment_bytes)
        segment_array, _ = sf.read(segment_buffer)
        audio_arrays.append(segment_array)
    
    combined_audio = np.concatenate(audio_arrays)
    combined_bytes = audio_to_bytes(combined_audio, 'wav')
    
    # Clean up
    del audio_arrays, combined_audio
    gc.collect()
    
    return combined_bytes

def _combine_audio_segments(audio_segments: List[bytes], format: str) -> bytes:
    """Combine audio segments of any format.
    
    Args:
        audio_segments: List of audio byte segments
        format: Audio format
        
    Returns:
        Combined audio bytes
    """
    if len(audio_segments) == 1:
        return audio_segments[0]
    
    from pydub import AudioSegment as AS
    
    # Load all segments
    audio_objects = []
    for segment_bytes in audio_segments:
        segment_buffer = io.BytesIO(segment_bytes)
        segment_audio = AS.from_file(segment_buffer, format=format)
        audio_objects.append(segment_audio)
    
    # Combine all segments
    combined_audio = audio_objects[0]
    for audio_obj in audio_objects[1:]:
        combined_audio += audio_obj
    
    # Export to bytes
    output_buffer = io.BytesIO()
    if format.lower() == 'mp3':
        combined_audio.export(output_buffer, format='mp3', bitrate='192k')
    elif format.lower() == 'opus':
        combined_audio.export(output_buffer, format='opus', codec='libopus')
    else:
        combined_audio.export(output_buffer, format=format)
    
    output_buffer.seek(0)
    result_bytes = output_buffer.getvalue()
    
    # Clean up
    del audio_objects, combined_audio
    gc.collect()
    
    return result_bytes

def _prepare_voice_for_synthesis(voice_spec: Union[str, Dict[str, float]], lang_pipeline: KPipeline):
    """Prepare voice for synthesis - either load single voice or create blended voice.
    
    Args:
        voice_spec: Either voice name string or dictionary of voice weights
        lang_pipeline: KPipeline instance
        
    Returns:
        Voice tensor ready for synthesis
        
    Raises:
        ValueError: If voice preparation fails
    """
    if isinstance(voice_spec, str):
        # Single voice - return voice name for KPipeline
        return voice_spec
    else:
        # Voice blending
        if not voice_spec:
            raise ValueError("Voice weights cannot be empty")
        
        # Normalize weights to sum to 1.0
        total_weight = sum(voice_spec.values())
        if total_weight <= 0:
            raise ValueError("Total voice weights must be positive")
        
        normalized_weights = {voice: weight / total_weight for voice, weight in voice_spec.items()}
        
        # Load and blend voice tensors
        blended_voice = None
        for voice_name, weight in normalized_weights.items():
            voice_tensor = lang_pipeline.load_voice(voice_name)
            if blended_voice is None:
                blended_voice = weight * voice_tensor
            else:
                blended_voice += weight * voice_tensor
        
        return blended_voice

def _core_synthesize(
    text: str,
    voice_spec: Union[str, Dict[str, float]],
    speed: float,
    format: str = "wav",
    lang_code: str = "a",
    include_captions: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE
) -> Union[bytes, SynthesisResult]:
    """Core synthesis engine that handles both single and blended voices.
    
    Args:
        text: Input text to synthesize
        voice_spec: Voice name string or dictionary of voice weights
        speed: Speech speed multiplier
        format: Audio format (wav, mp3, etc.)
        lang_code: Language code
        include_captions: If True, returns SynthesisResult with word timings
        chunk_size: Size for text chunking
        
    Returns:
        bytes if include_captions=False, SynthesisResult if include_captions=True
        
    Raises:
        ValueError: For input validation errors
        RuntimeError: For synthesis failures
    """
    if not text.strip():
        raise ValueError("Text cannot be empty")
    
    try:
        # Get pipeline for the specified language
        lang_pipeline = get_pipeline(lang_code)
        
        # Prepare voice (single or blended)
        prepared_voice = _prepare_voice_for_synthesis(voice_spec, lang_pipeline)
        
        # Use intelligent text segmentation for better results
        text_chunks = text_segmentation(text, chunk_size)
        voice_type = "blended" if isinstance(voice_spec, dict) else "single"
        
        # Check if we should use batch processing for very long texts
        total_chars = len(text)
        use_batch_processing = should_use_batch_processing(text)
        
        logger.info(
            f"Processing {total_chars:,} characters in {len(text_chunks)} chunks "
            f"for {voice_type} voice (lang: {lang_code}, batch: {use_batch_processing})"
        )
        
        # Memory management for long texts
        if use_batch_processing:
            return _batch_synthesize(
                text_chunks, voice_spec, prepared_voice, speed, format, 
                lang_code, include_captions, lang_pipeline
            )
        
        all_audio_data = []
        all_word_timings = [] if include_captions else None
        chunk_time_offset = 0.0 if include_captions else 0.0
        
        for chunk_idx, chunk in enumerate(text_chunks):
            logger.debug(f"Processing {voice_type} chunk {chunk_idx + 1}/{len(text_chunks)}: {chunk[:50]}...")
            
            if include_captions:
                chunk_audio_data = []
                chunk_word_timings = []
            
            # Use prepared voice (string for single voice, tensor for blended)
            if isinstance(prepared_voice, str):
                generator = lang_pipeline(chunk, prepared_voice, speed)
            else:
                generator = lang_pipeline(chunk, voice=prepared_voice, speed=speed)
                
            for result in generator:
                audio = result.audio.cpu().numpy()
                logger.debug(f"Generated sub-chunk for {voice_type} chunk {chunk_idx + 1}: {len(audio)} samples")
                
                if include_captions:
                    # Extract timing information
                    sub_chunk_timings = extract_word_timings(result, chunk_time_offset)
                    chunk_word_timings.extend(sub_chunk_timings)
                    chunk_audio_data.append(audio)
                    
                    # Update time offset for next sub-chunk
                    chunk_time_offset += calculate_audio_duration(audio)
                else:
                    all_audio_data.append(audio)
            
            # Handle audio data for captions
            if include_captions and chunk_audio_data:
                chunk_combined_audio = np.concatenate(chunk_audio_data) if len(chunk_audio_data) > 1 else chunk_audio_data[0]
                all_audio_data.append(chunk_combined_audio)
                all_word_timings.extend(chunk_word_timings)
        
        # Concatenate all audio data efficiently
        if not all_audio_data:
            raise RuntimeError("No audio data generated")
        
        combined_audio = np.concatenate(all_audio_data) if len(all_audio_data) > 1 else all_audio_data[0]
        
        logger.info(f"Combined {len(all_audio_data)} {voice_type} audio segments: {len(combined_audio)} total samples")
        
        if include_captions:
            total_duration = calculate_audio_duration(combined_audio)
            audio_bytes = audio_to_bytes(combined_audio, format)
            logger.info(f"Generated {len(all_word_timings)} words with timing information for {voice_type} voice")
            
            return SynthesisResult(
                word_timings=all_word_timings,
                audio_bytes=audio_bytes,
                total_duration=total_duration,
                sample_rate=AUDIO_SAMPLE_RATE
            )
        else:
            return audio_to_bytes(combined_audio, format)
            
    except Exception as e:
        voice_desc = "voice blend" if isinstance(voice_spec, dict) else "single voice"
        raise RuntimeError(f"{voice_desc.title()} synthesis failed: {str(e)}") from e

def synthesize(
    text: str, 
    speaker: str, 
    speed: float, 
    format: str = "wav", 
    lang_code: str = "a", 
    include_captions: bool = False
) -> Union[bytes, SynthesisResult]:
    """Generate text-to-speech audio and return as bytes with automatic text chunking
    
    Args:
        text: Input text to synthesize
        speaker: Voice name to use
        speed: Speech speed multiplier
        format: Audio format (wav, mp3, etc.)
        lang_code: Language code
        include_captions: If True, returns SynthesisResult with word timings; if False, returns bytes
    
    Returns:
        bytes if include_captions=False, SynthesisResult if include_captions=True
    """
    return _core_synthesize(text, speaker, speed, format, lang_code, include_captions)

def synthesize_with_voice_blend(text: str, voice_weights: Dict[str, float], speed: float, format: str = "wav", lang_code: str = "a", include_captions: bool = False) -> Union[bytes, SynthesisResult]:
    """Generate text-to-speech audio with blended voices and return as bytes with automatic text chunking
    
    Args:
        text: Input text to synthesize
        voice_weights: Dictionary mapping voice names to blend weights
        speed: Speech speed multiplier
        format: Audio format (wav, mp3, etc.)
        lang_code: Language code
        include_captions: If True, returns SynthesisResult with word timings; if False, returns bytes
    
    Returns:
        bytes if include_captions=False, SynthesisResult if include_captions=True
    """
    return _core_synthesize(text, voice_weights, speed, format, lang_code, include_captions, STREAMING_CHUNK_SIZE)

def _core_streaming_synthesize(
    text: str,
    voice_spec: Union[str, Dict[str, float]],
    speed: float,
    format: str = "wav",
    lang_code: str = "a",
    include_captions: bool = False
) -> Union[Iterator[bytes], Iterator[StreamingCaptionChunk]]:
    """Core streaming synthesis engine that handles both single and blended voices.
    
    Args:
        text: Input text to synthesize
        voice_spec: Voice name string or dictionary of voice weights
        speed: Speech speed multiplier
        format: Audio format (wav, mp3, etc.)
        lang_code: Language code
        include_captions: If True, returns Iterator[StreamingCaptionChunk]
        
    Returns:
        Iterator[bytes] if include_captions=False, Iterator[StreamingCaptionChunk] if include_captions=True
        
    Raises:
        ValueError: For input validation errors
        RuntimeError: For synthesis failures
    """
    if not text.strip():
        raise ValueError("Text cannot be empty")
    
    try:
        # Get pipeline for the specified language
        lang_pipeline = get_pipeline(lang_code)
        
        # Prepare voice (single or blended)
        prepared_voice = _prepare_voice_for_synthesis(voice_spec, lang_pipeline)
        
        # Use intelligent text segmentation optimized for streaming
        streaming_chunk_size = 600  # Smaller chunks for streaming
        text_chunks = text_segmentation(text, streaming_chunk_size)
        voice_type = "blended" if isinstance(voice_spec, dict) else "single"
        
        total_chars = len(text)
        logger.info(
            f"Streaming {total_chars:,} characters in {len(text_chunks)} chunks "
            f"for {voice_type} voice (lang: {lang_code})"
        )
        
        chunk_counter = 0
        chunk_time_offset = 0.0 if include_captions else 0.0
        
        for chunk_idx, chunk in enumerate(text_chunks):
            logger.debug(f"Streaming {voice_type} chunk {chunk_idx + 1}/{len(text_chunks)}: {chunk[:50]}...")
            
            # Use prepared voice (string for single voice, tensor for blended)
            if isinstance(prepared_voice, str):
                generator = lang_pipeline(chunk, prepared_voice, speed)
            else:
                generator = lang_pipeline(chunk, voice=prepared_voice, speed=speed)
                
            for result in generator:
                chunk_counter += 1
                audio = result.audio.cpu().numpy()
                logger.debug(f"Streaming {voice_type} sub-chunk {chunk_counter}: {result.graphemes} -> {len(audio)} samples")
                
                if include_captions:
                    # Extract timing information for this sub-chunk
                    sub_chunk_timings = extract_word_timings(result, chunk_time_offset)
                    
                    # Convert audio to bytes
                    if format.lower() == "wav":
                        if chunk_counter == 1:
                            # Send WAV header first, then raw audio data
                            yield StreamingCaptionChunk(
                                word_timings=[],
                                audio_data=create_wav_streaming_header(),
                                chunk_number=chunk_counter,
                                is_final=False
                            )
                            chunk_counter += 1
                            
                        audio_bytes = audio_to_raw_bytes(audio)
                    else:
                        # For non-WAV formats, each chunk needs complete file headers
                        audio_bytes = audio_to_bytes(audio, format)
                    
                    # Update time offset for next sub-chunk
                    chunk_time_offset += calculate_audio_duration(audio)
                    
                    # Determine if this is the final chunk (approximate)
                    is_final = chunk_idx == len(text_chunks) - 1
                    
                    yield StreamingCaptionChunk(
                        word_timings=sub_chunk_timings,
                        audio_data=audio_bytes,
                        chunk_number=chunk_counter,
                        is_final=is_final
                    )
                else:
                    # Audio-only streaming
                    if format.lower() == "wav":
                        if chunk_counter == 1:
                            # Send WAV header first, then raw audio data
                            yield create_wav_streaming_header()
                            yield audio_to_raw_bytes(audio)
                        else:
                            # All subsequent chunks are raw audio data
                            yield audio_to_raw_bytes(audio)
                    else:
                        # For non-WAV formats, each chunk needs complete file headers
                        yield audio_to_bytes(audio, format)
                        
    except Exception as e:
        voice_desc = "voice blend" if isinstance(voice_spec, dict) else "single voice"
        raise RuntimeError(f"{voice_desc.title()} streaming synthesis failed: {str(e)}") from e

def synthesize_streaming(text: str, speaker: str, speed: float, format: str = "wav", lang_code: str = "a", include_captions: bool = False) -> Union[Iterator[bytes], Iterator[StreamingCaptionChunk]]:
    """Generate text-to-speech audio in streaming chunks with automatic text chunking
    
    Args:
        text: Input text to synthesize
        speaker: Voice name to use
        speed: Speech speed multiplier
        format: Audio format (wav, mp3, etc.)
        lang_code: Language code
        include_captions: If True, returns Iterator[StreamingCaptionChunk] with word timings; if False, returns Iterator[bytes]
    
    Returns:
        Iterator[bytes] if include_captions=False, Iterator[StreamingCaptionChunk] if include_captions=True
    """
    return _core_streaming_synthesize(text, speaker, speed, format, lang_code, include_captions)

def synthesize_streaming_with_voice_blend(text: str, voice_weights: Dict[str, float], speed: float, format: str = "wav", lang_code: str = "a", include_captions: bool = False) -> Union[Iterator[bytes], Iterator[StreamingCaptionChunk]]:
    """Generate text-to-speech audio with blended voices in streaming chunks with automatic text chunking
    
    Args:
        text: Input text to synthesize
        voice_weights: Dictionary mapping voice names to blend weights
        speed: Speech speed multiplier
        format: Audio format (wav, mp3, etc.)
        lang_code: Language code
        include_captions: If True, returns Iterator[StreamingCaptionChunk] with word timings; if False, returns Iterator[bytes]
    
    Returns:
        Iterator[bytes] if include_captions=False, Iterator[StreamingCaptionChunk] if include_captions=True
    """
    return _core_streaming_synthesize(text, voice_weights, speed, format, lang_code, include_captions)