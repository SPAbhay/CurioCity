# backend/tts_utils.py
import os
import uuid # For unique filenames, as in your example
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings # For detailed voice customization
from fastapi.concurrency import run_in_threadpool
import asyncio

# --- ElevenLabs Client Initialization ---
# It's often better to initialize the client once if the API key doesn't change.
# However, for safety within threadpool and potential key changes via .env,
# initializing it inside the blocking function or ensuring thread-safety is key.
# For now, let's keep initialization inside _generate_and_save_audio_blocking
# to ensure it picks up the env var correctly in the thread.

def _generate_and_save_audio_blocking(
    text: str, 
    voice_id: str, 
    model_id: str, 
    full_output_path: str,
    voice_settings: VoiceSettings = None # Optional VoiceSettings
):
    """
    Blocking function to generate audio using ElevenLabs and save it by writing chunks.
    """
    try:
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            print("Critical Error: ELEVENLABS_API_KEY not found in environment.")
            return None
            
        client = ElevenLabs(api_key=api_key)
        
        print(f"Requesting audio from ElevenLabs. Voice: {voice_id}, Model: {model_id}, Text: '{text[:50]}...'")
        
        response_audio_stream = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id=model_id,
            voice_settings=voice_settings # Pass VoiceSettings if provided
            # output_format="mp3_44100_128" # This is often default for MP3 or handled by file extension
        )
        
        with open(full_output_path, "wb") as f:
            for chunk in response_audio_stream:
                if chunk:
                    f.write(chunk)
        
        if os.path.exists(full_output_path) and os.path.getsize(full_output_path) > 0:
            print(f"ElevenLabs audio content successfully saved to {full_output_path}")
            return full_output_path
        else:
            print(f"Failed to save ElevenLabs audio file or file is empty: {full_output_path}")
            # Attempt to remove empty file if it exists
            if os.path.exists(full_output_path):
                os.remove(full_output_path)
            return None
            
    except Exception as e:
        print(f"Error communicating with ElevenLabs or saving audio: {e}")
        import traceback
        traceback.print_exc()
        return None

async def text_to_speech_elevenlabs_async(
    text: str, 
    output_filename_base: str = "podcast_segment", # Base name, will add UUID and extension
    output_dir: str = "audio_outputs",
    voice_id: str = "pNInz6obpgDQGcFmaJgB",  # Default voice (Adam)
    model_id: str = "eleven_multilingual_v2", # Default model
    voice_settings_dict: dict = None # Optional dict for VoiceSettings
):
    """
    Converts text to speech using ElevenLabs API and saves to a unique MP3 file asynchronously.
    Returns the full path to the generated audio file, or None if an error occurs.
    """
    if not os.getenv("ELEVENLABS_API_KEY"):
        print("Warning: ELEVENLABS_API_KEY not found in environment. TTS will likely fail.")
        from dotenv import load_dotenv
        load_dotenv()
        if not os.getenv("ELEVENLABS_API_KEY"):
             print("Critical Error: ELEVENLABS_API_KEY is still not found after attempting to load .env.")
             return None

    os.makedirs(output_dir, exist_ok=True)
    # Generate a unique filename using UUID to prevent overwrites
    unique_filename = f"{output_filename_base}_{uuid.uuid4()}.mp3"
    full_output_path = os.path.join(output_dir, unique_filename)
    
    print(f"Dispatching ElevenLabs TTS generation for {full_output_path} to threadpool...")
    
    current_voice_settings = None
    if voice_settings_dict:
        try:
            current_voice_settings = VoiceSettings(**voice_settings_dict)
        except Exception as e:
            print(f"Warning: Could not parse voice_settings_dict: {e}. Using default voice settings.")

    result_path = await run_in_threadpool(
        _generate_and_save_audio_blocking, 
        text, 
        voice_id,
        model_id,
        full_output_path,
        current_voice_settings # Pass the VoiceSettings object
    )
    
    if result_path:
        print(f"Async ElevenLabs TTS task completed. Audio file: {result_path}")
        return unique_filename # Return just the filename for the API response
    else:
        print(f"Async ElevenLabs TTS task completed. Audio file generation failed for {full_output_path}.")
        return None

# --- Test function ---
async def main_test():
    from dotenv import load_dotenv
    load_dotenv()

    sample_text_casey = "Hello, this is Curious Casey testing my voice!"
    sample_text_finn = "This is Factual Finn, providing a test statement."
    
    print("Starting ElevenLabs TTS test for Casey...")
    casey_filename = await text_to_speech_elevenlabs_async(
        text=sample_text_casey,
        output_filename_base="test_casey",
        voice_id="F2OOWcJMWhX8wCsyT0oR", # Curious Casey's Voice ID
        model_id="eleven_turbo_v2_5", # Example model from your snippet
        voice_settings_dict={"stability": 0.5, "similarity_boost": 0.75, "style": 0.0, "use_speaker_boost": True} # Example settings
    )
    if casey_filename:
        print(f"Casey test successful! Audio filename: {casey_filename}")
    else:
        print("Casey test failed.")

    print("\nStarting ElevenLabs TTS test for Finn...")
    finn_filename = await text_to_speech_elevenlabs_async(
        text=sample_text_finn,
        output_filename_base="test_finn",
        voice_id="IFEvkitzF8OoHeggkJUu", # Factual Finn's Voice ID
        model_id="eleven_multilingual_v2" # Different model for Finn, or same
    )
    if finn_filename:
        print(f"Finn test successful! Audio filename: {finn_filename}")
    else:
        print("Finn test failed.")

if __name__ == "__main__":
    asyncio.run(main_test())