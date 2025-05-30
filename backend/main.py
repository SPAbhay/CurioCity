from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from graph_orchestrator import app_graph, AgentState

from pydub import AudioSegment # For concatinating audio
from tts_utils import text_to_speech_elevenlabs_async as text_to_speech_file
import time

from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
OLLAMA_FACTUAL_FINN_MODEL_NAME = "factual-finn"
OLLAMA_CURIOUS_CASEY_MODEL_NAME = "curious-casey" # New model name for Casey
OLLAMA_BASE_URL = "http://localhost:11434"

# ElevenLabs
CASEY_VOICE_ID = "F2OOWcJMWhX8wCsyT0oR" 
FINN_VOICE_ID = "IFEvkitzF8OoHeggkJUu" 
CASEY_TTS_MODEL_ID = "eleven_turbo_v2_5" 
FINN_TTS_MODEL_ID = "eleven_multilingual_v2" 

# Alpaca prompt template string (used for Factual Finn)
alpaca_prompt_template_str = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""

# Global variables for the LangChain ChatOllama instances
factual_finn_llm = None
curious_casey_llm = None # New LLM instance for Casey

# --- FastAPI Lifespan for Ollama Client Initialization ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global factual_finn_llm, curious_casey_llm
    print(f"Initializing ChatOllama for Factual Finn: {OLLAMA_FACTUAL_FINN_MODEL_NAME}...")
    try:
        factual_finn_llm = ChatOllama(
            model=OLLAMA_FACTUAL_FINN_MODEL_NAME,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
        )
        print("Factual Finn ChatOllama client initialized.")
    except Exception as e:
        print(f"Error initializing Factual Finn ChatOllama client: {e}")
        factual_finn_llm = None

    print(f"Initializing ChatOllama for Curious Casey: {OLLAMA_CURIOUS_CASEY_MODEL_NAME}...")
    try:
        curious_casey_llm = ChatOllama(
            model=OLLAMA_CURIOUS_CASEY_MODEL_NAME,
            base_url=OLLAMA_BASE_URL,
            temperature=0.7, 
            stop=["\n", "<|eot_id|>", "<|end_of_text|>", "The knowledgeable AI just said:"]
        )
        print("Curious Casey ChatOllama client initialized.")
    except Exception as e:
        print(f"Error initializing Curious Casey ChatOllama client: {e}")
        curious_casey_llm = None
    
    yield

    print("FastAPI app shutting down.")

app = FastAPI(lifespan=lifespan)

# --- Pydantic Models for Request Bodies ---
class ExplanationRequest(BaseModel):
    information_text: str
    instruction: str = "Explain the following information in a direct and factual style, focusing on key points."

class FollowUpRequest(BaseModel): # New Pydantic model for Casey's input
    previous_statement: str
    # Casey's instruction is baked into its fine-tuning and system prompt,
    # but we could allow overriding it here if needed in the future.
    # instruction: str = "Given the previous statement from the knowledgeable AI, ask a relevant and engaging follow-up question..."

class StartPodcastRequest(BaseModel):
    initial_information: str

# --- API Endpoints ---
@app.get("/")
async def root():
    finn_status = "Ready" if factual_finn_llm else "Failed to Initialize"
    casey_status = "Ready" if curious_casey_llm else "Failed to Initialize"
    return {
        "message": "AI Podcast Backend is running!",
        "factual_finn_status": finn_status,
        "curious_casey_status": casey_status
    }

@app.post("/explain") # For Factual Finn
async def get_explanation(request: ExplanationRequest):
    if not factual_finn_llm:
        raise HTTPException(status_code=503, detail="Factual Finn (Ollama) service not initialized or unavailable.")

    print(f"Received request for Factual Finn: {request.information_text[:50]}...")
    formatted_prompt = alpaca_prompt_template_str.format(
        instruction=request.instruction,
        input_text=request.information_text
    )
    try:
        print(f"Sending prompt to Factual Finn (Ollama):\n{formatted_prompt}")
        response = await factual_finn_llm.ainvoke(formatted_prompt)
        explanation_text = response.content
        print(f"Received response from Factual Finn (Ollama): {explanation_text[:100]}...")
        return {"explanation": explanation_text.strip()}
    except Exception as e:
        print(f"Error during Factual Finn (Ollama) API call: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get explanation from Factual Finn (Ollama): {str(e)}")

@app.post("/ask_follow_up") # New endpoint for Curious Casey
async def ask_follow_up_question(request: FollowUpRequest):
    if not curious_casey_llm:
        raise HTTPException(status_code=503, detail="Curious Casey (Ollama) service not initialized or unavailable.")

    print(f"Received request for Curious Casey: {request.previous_statement[:50]}...")
    
    # For Curious Casey, the input to its fine-tuning was:
    # "The knowledgeable AI just said: '[previous_statement]'"
    # This full string is what Casey expects as the user's prompt content.
    # The instruction "Given the previous statement..." is part of Casey's fine-tuning
    # and also reinforced by its Ollama system prompt.
    casey_prompt_input = f"The knowledgeable AI just said: '{request.previous_statement}'"
    
    try:
        print(f"Sending prompt to Curious Casey (Ollama):\n{casey_prompt_input}")
        # We send this directly as the prompt to Casey, as its fine-tuning
        # expects this as the "user" part of the conversation.
        response = await curious_casey_llm.ainvoke(casey_prompt_input)
        question_text = response.content
        
        print(f"Received question from Curious Casey (Ollama): {question_text[:100]}...")
        return {"follow_up_question": question_text.strip()}
    except Exception as e:
        print(f"Error during Curious Casey (Ollama) API call: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get question from Curious Casey (Ollama): {str(e)}")
    
@app.post("/initiate_podcast_flow", response_model=AgentState)
async def initiate_podcast_flow(request: StartPodcastRequest): # StartPodcastRequest should be defined
    if not app_graph:
        raise HTTPException(status_code=500, detail="LangGraph application not initialized.")
    if not factual_finn_llm or not curious_casey_llm: # Check if Ollama clients are ready
         raise HTTPException(status_code=503, detail="One or more Ollama LLM clients are not ready.")

    print(f"Received request to initiate podcast flow with: {request.initial_information[:100]}...")

    initial_graph_input = AgentState(
        initial_user_input=request.initial_information,
        casey_question="",
        finn_explanation="",
        conversation_history=[],
        current_turn=0,
        generated_audio_file=None 
    )

    try:
        print("Invoking LangGraph app_graph...")
        final_state_dict = await app_graph.ainvoke(initial_graph_input, {"recursion_limit": 10})
        print(f"LangGraph execution complete. Final state from graph: {final_state_dict}")

        # --- TTS Generation with Distinct Voices ---
        conversation_history = final_state_dict.get("conversation_history", [])
        generated_audio_segments_paths = [] # To store paths of individual audio files
        final_podcast_filename_only = None

        if conversation_history:
            audio_output_dir = "audio_outputs" # Ensure this directory exists
            os.makedirs(audio_output_dir, exist_ok=True)

            # Generate unique base for segment filenames for this request
            request_timestamp = int(time.time())

            for i, turn_text in enumerate(conversation_history):
                speaker = ""
                text_to_speak = ""
                current_voice_id = ""
                current_model_id = ""

                if turn_text.startswith("Curious Casey:"):
                    speaker = "Curious Casey"
                    text_to_speak = turn_text.replace("Curious Casey: ", "", 1).strip()
                    current_voice_id = CASEY_VOICE_ID
                    current_model_id = CASEY_TTS_MODEL_ID # Or a common model
                elif turn_text.startswith("Factual Finn:"):
                    speaker = "Factual Finn"
                    text_to_speak = turn_text.replace("Factual Finn: ", "", 1).strip()
                    current_voice_id = FINN_VOICE_ID
                    current_model_id = FINN_TTS_MODEL_ID # Or a common model
                elif turn_text.startswith("Initial Topic:"):
                    # Optionally, you could have a "Narrator" voice for the initial topic
                    # For now, let's skip TTS for the "Initial Topic" or assign a default.
                    # Or, have Casey "read" it as an intro.
                    # Let's try having Casey read the intro topic in her voice for this example.
                    speaker = "Narrator (Casey)" # Or just Casey
                    text_to_speak = turn_text.replace("Initial Topic: ", "", 1).strip()
                    current_voice_id = CASEY_VOICE_ID # Use Casey's voice for the intro topic
                    current_model_id = CASEY_TTS_MODEL_ID
                else:
                    print(f"Skipping TTS for unknown speaker in turn: {turn_text}")
                    continue

                if not text_to_speak: # Skip empty turns
                    continue

                segment_filename_base = f"segment_{request_timestamp}_{i}_{speaker.replace(' ', '_').lower()}"

                print(f"Converting turn {i+1} ({speaker}) to speech: '{text_to_speak[:50]}...'")
                segment_path = await text_to_speech_file( # This is your ElevenLabs async function
                    text=text_to_speak,
                    output_filename_base=segment_filename_base,
                    output_dir=audio_output_dir,
                    voice_id=current_voice_id,
                    model_id=current_model_id
                    # voice_settings_dict can be added here if you want per-speaker settings
                )
                if segment_path: # text_to_speech_file now returns just the filename
                    generated_audio_segments_paths.append(os.path.join(audio_output_dir, segment_path))
                else:
                    print(f"Failed to generate audio for segment: {speaker} - {text_to_speak[:50]}...")

            # Concatenate audio segments if we have any
            if generated_audio_segments_paths:
                print("Concatenating audio segments...")
                combined_audio = AudioSegment.empty()
                for segment_path in generated_audio_segments_paths:
                    try:
                        sound_segment = AudioSegment.from_mp3(segment_path) # Assuming MP3 from ElevenLabs
                        combined_audio += sound_segment
                    except Exception as e:
                        print(f"Error loading segment {segment_path} for concatenation: {e}")
                        continue # Skip problematic segment

                if len(combined_audio) > 0:
                    final_podcast_filename_only = f"podcast_full_{request_timestamp}.mp3"
                    final_podcast_full_path = os.path.join(audio_output_dir, final_podcast_filename_only)
                    try:
                        combined_audio.export(final_podcast_full_path, format="mp3")
                        print(f"Final concatenated podcast audio saved: {final_podcast_full_path}")
                        final_state_dict["generated_audio_file"] = final_podcast_filename_only
                    except Exception as e:
                        print(f"Error exporting combined audio: {e}")
                        final_state_dict["generated_audio_file"] = None
                else:
                    print("No valid audio segments to combine.")
                    final_state_dict["generated_audio_file"] = None
            else:
                print("No audio segments were generated.")
                final_state_dict["generated_audio_file"] = None
        else:
            print("No conversation history to convert to speech.")
            final_state_dict["generated_audio_file"] = None
        # --- End TTS Generation ---

        return final_state_dict

    except Exception as e:
        print(f"Error during LangGraph or TTS execution: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error in podcast flow: {str(e)}")
    
@app.get("/get_podcast_audio/{filename}")
async def get_podcast_audio(filename: str):
    # Ensure the audio_outputs directory is correctly referenced
    # (it should be relative to where main.py is running from, or use absolute paths)
    file_path = os.path.join("audio_outputs", filename) 
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='audio/mpeg', filename=filename)
    else:
        print(f"Audio file not found at: {file_path}")
        raise HTTPException(status_code=404, detail="Audio file not found.")