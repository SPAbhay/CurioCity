from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
import uuid
import time
import traceback # For detailed error logging

from typing import Optional, List
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from graph_orchestrator import app_graph, AgentState # Import your compiled graph and state definition
from dotenv import load_dotenv

from fastapi.responses import FileResponse
from tts_utils import text_to_speech_elevenlabs_async as text_to_speech_file # Your ElevenLabs TTS function
from pydub import AudioSegment # For concatenating audio

load_dotenv() 

# --- Configuration ---
OLLAMA_FACTUAL_FINN_MODEL_NAME = "factual-finn"
OLLAMA_CURIOUS_CASEY_MODEL_NAME = "curious-casey"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

CASEY_VOICE_ID = os.getenv("CASEY_VOICE_ID", "F2OOWcJMWhX8wCsyT0oR")
FINN_VOICE_ID = os.getenv("FINN_VOICE_ID", "IFEvkitzF8OoHeggkJUu")
NARRATOR_VOICE_ID = os.getenv("NARRATOR_VOICE_ID", CASEY_VOICE_ID) # Defaulting to Casey for intro/topic

CASEY_TTS_MODEL_ID = os.getenv("CASEY_TTS_MODEL_ID", "eleven_turbo_v2_5")
FINN_TTS_MODEL_ID = os.getenv("FINN_TTS_MODEL_ID", "eleven_multilingual_v2")
NARRATOR_TTS_MODEL_ID = os.getenv("NARRATOR_TTS_MODEL_ID", CASEY_TTS_MODEL_ID)

AUDIO_OUTPUT_DIR = "audio_outputs"

alpaca_prompt_template_str = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""

# Global LLM clients
factual_finn_llm = None
curious_casey_llm = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global factual_finn_llm, curious_casey_llm, app_graph
    os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True) # Ensure audio output dir exists
    
    print(f"Initializing ChatOllama for Factual Finn: {OLLAMA_FACTUAL_FINN_MODEL_NAME}...")
    try:
        factual_finn_llm = ChatOllama(
            model=OLLAMA_FACTUAL_FINN_MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0,
            stop=["<|eot_id|>", "<|end_of_text|>"])
        print("Factual Finn ChatOllama client initialized.")
    except Exception as e:
        print(f"Error initializing Factual Finn ChatOllama client: {e}\n{traceback.format_exc()}")
    
    print(f"Initializing ChatOllama for Curious Casey: {OLLAMA_CURIOUS_CASEY_MODEL_NAME}...")
    try:
        curious_casey_llm = ChatOllama(
            model=OLLAMA_CURIOUS_CASEY_MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0.7,
            stop=["<|eot_id|>", "<|end_of_text|>", "\n\n", "The knowledgeable AI just said:"])
        print("Curious Casey ChatOllama client initialized.")
    except Exception as e:
        print(f"Error initializing Curious Casey ChatOllama client: {e}\n{traceback.format_exc()}")

    if app_graph:
        print("LangGraph app_graph loaded from orchestrator.")
    else:
        print("CRITICAL Error: LangGraph app_graph not loaded! Check graph_orchestrator.py.")
    yield
    print("FastAPI app shutting down.")

app = FastAPI(lifespan=lifespan)

# --- Pydantic Models ---
class StartPodcastRequest(BaseModel):
    initial_information: str
    generate_audio: bool = False
    user_provided_themes: Optional[List[str]] = None

class SubmitDoubtRequest(BaseModel):
    user_doubt_text: str
    generate_audio: bool = False

class ExplanationRequest(BaseModel):
    information_text: str
    instruction: str

class FollowUpRequest(BaseModel):
    previous_statement: str
    
class GenerateThemesRequest(BaseModel):
    topic_info: str 
    generation_prompt: str 
    
class GuidingThemesOutput(BaseModel):
    themes: List[str] = Field(description="A list of 3 to 4 key open-ended questions or themes for the podcast.")

# --- Helper Function for Multi-Voice TTS ---
async def _generate_podcast_audio(conversation_history: list, request_timestamp: int, filename_prefix: str) -> Optional[str]:
    generated_audio_segments_paths = []
    final_podcast_filename_only = None

    if not conversation_history:
        print("No conversation history to convert to speech.")
        return None

    for i, turn_text in enumerate(conversation_history):
        speaker_label = ""
        text_to_speak = ""
        current_voice_id = ""
        current_model_id = ""

        if turn_text.startswith("Curious Casey:"):
            speaker_label = "Casey"
            text_to_speak = turn_text.replace("Curious Casey: ", "", 1).strip()
            current_voice_id = CASEY_VOICE_ID
            current_model_id = CASEY_TTS_MODEL_ID
        elif turn_text.startswith("Factual Finn (addressing doubt):"):
            speaker_label = "Finn_Doubt"
            text_to_speak = turn_text.replace("Factual Finn (addressing doubt): ", "", 1).strip()
            current_voice_id = FINN_VOICE_ID
            current_model_id = FINN_TTS_MODEL_ID
        elif turn_text.startswith("Factual Finn:"):
            speaker_label = "Finn"
            text_to_speak = turn_text.replace("Factual Finn: ", "", 1).strip()
            current_voice_id = FINN_VOICE_ID
            current_model_id = FINN_TTS_MODEL_ID
        elif turn_text.startswith("Initial Topic:"):
            speaker_label = "Narrator"
            text_to_speak = turn_text.replace("Initial Topic: ", "", 1).strip()
            current_voice_id = NARRATOR_VOICE_ID
            current_model_id = NARRATOR_TTS_MODEL_ID
        elif turn_text.startswith("User Doubt:"):
            # Decide if you want to TTS the user's doubt.
            # For now, let's skip TTSing the user's doubt text directly.
            # The context is given to Finn, and Finn's answer will be voiced.
            print(f"Skipping TTS for 'User Doubt' text: {turn_text[:50]}...")
            continue
        else:
            print(f"Skipping TTS for unrecognized turn format: {turn_text[:50]}...")
            continue

        if not text_to_speak:
            continue
        
        segment_filename_base = f"segment_{request_timestamp}_{i}_{speaker_label}"
        
        print(f"Converting turn {i+1} ({speaker_label}) for TTS: '{text_to_speak[:50]}...'")
        segment_filename = await text_to_speech_file(
            text=text_to_speak, 
            output_filename_base=segment_filename_base,
            output_dir=AUDIO_OUTPUT_DIR,
            voice_id=current_voice_id,
            model_id=current_model_id
        )
        if segment_filename:
            generated_audio_segments_paths.append(os.path.join(AUDIO_OUTPUT_DIR, segment_filename))
        else:
            print(f"Failed to generate audio for segment: {speaker_label} - {text_to_speak[:50]}...")
    
    if generated_audio_segments_paths:
        print("Concatenating audio segments...")
        combined_audio = AudioSegment.empty()
        for segment_path in generated_audio_segments_paths:
            if os.path.exists(segment_path):
                try:
                    sound_segment = AudioSegment.from_mp3(segment_path)
                    combined_audio += sound_segment
                except Exception as e:
                    print(f"Error loading audio segment {segment_path} for concatenation: {e}")
            else:
                print(f"Audio segment file not found, skipping: {segment_path}")
        
        if len(combined_audio) > 0:
            final_podcast_filename_only = f"{filename_prefix}_{request_timestamp}.mp3"
            final_podcast_full_path = os.path.join(AUDIO_OUTPUT_DIR, final_podcast_filename_only)
            try:
                combined_audio.export(final_podcast_full_path, format="mp3")
                print(f"Final concatenated podcast audio saved: {final_podcast_full_path}")
                return final_podcast_filename_only
            except Exception as e:
                print(f"Error exporting combined audio: {e}\n{traceback.format_exc()}")
        else:
            print("No valid audio segments to combine.")
    else:
        print("No audio segments were generated.")
    return None

# --- API Endpoints ---
@app.get("/")
async def root():
    finn_status = "Ready" if factual_finn_llm else "Failed"
    casey_status = "Ready" if curious_casey_llm else "Failed"
    graph_status = "Ready" if app_graph else "Failed"
    return {
        "message": "AI Podcast Backend is running!",
        "factual_finn_status": finn_status,
        "curious_casey_status": casey_status,
        "langgraph_status": graph_status
    }

@app.post("/explain", response_model=dict)
async def get_explanation_from_finn_model(request: ExplanationRequest):
    if not factual_finn_llm:
        raise HTTPException(status_code=503, detail="Factual Finn (Ollama) service not initialized.")
    try:
        full_prompt_for_finn = alpaca_prompt_template_str.format(
            instruction=request.instruction,
            input_text=request.information_text
        )
        response = await factual_finn_llm.ainvoke(full_prompt_for_finn)
        return {"explanation": response.content.strip()}
    except Exception as e:
        print(f"Error in /explain endpoint: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error generating explanation: {str(e)}")

@app.post("/ask_follow_up", response_model=dict)
async def get_question_from_casey_model(request: FollowUpRequest):
    if not curious_casey_llm:
        raise HTTPException(status_code=503, detail="Curious Casey (Ollama) service not initialized.")
    try:
        casey_direct_input = f"The knowledgeable AI just said: '{request.previous_statement}'"
        response = await curious_casey_llm.ainvoke(casey_direct_input)
        return {"follow_up_question": response.content.strip()}
    except Exception as e:
        print(f"Error in /ask_follow_up endpoint: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error generating question: {str(e)}")

@app.post("/initiate_podcast_flow", response_model=AgentState)
async def initiate_podcast_flow(request: StartPodcastRequest): # Updated to use new StartPodcastRequest
    if not app_graph:
        raise HTTPException(status_code=500, detail="LangGraph application not initialized.")
    if not factual_finn_llm or not curious_casey_llm:
         raise HTTPException(status_code=503, detail="One or more Ollama LLM clients are not ready.")

    conversation_thread_id = str(uuid.uuid4()) # Assuming uuid is imported
    print(f"Initiating podcast flow. Thread ID: {conversation_thread_id}, Generate Audio: {request.generate_audio}")
    if request.user_provided_themes:
        print(f"User provided themes: {request.user_provided_themes}")

    initial_graph_input = AgentState(
        initial_user_input=request.initial_information,
        casey_question="", 
        finn_explanation="", 
        conversation_history=[],
        current_turn=0, 
        generated_audio_file=None, 
        user_doubt=None,
        guiding_themes=request.user_provided_themes, # Pass user themes if provided, else None
        covered_themes=[] # Always initialize covered_themes as empty list
    )

    config = {"configurable": {"thread_id": conversation_thread_id}, "recursion_limit": 10}

    try:
        print(f"Invoking LangGraph app_graph for thread_id: {conversation_thread_id}...")
        final_state_dict = await app_graph.ainvoke(initial_graph_input, config=config) 
        print(f"LangGraph execution complete for thread_id: {conversation_thread_id}. Final state from graph: {final_state_dict}")

        # --- Conditional TTS Generation ---
        if request.generate_audio: # Check the toggle
            conversation_text_for_tts = "\n\n".join(final_state_dict.get("conversation_history", []))
            generated_audio_filename_only = None

            if conversation_text_for_tts:
                audio_output_dir = "audio_outputs"
                os.makedirs(audio_output_dir, exist_ok=True) # Ensure directory exists
                request_timestamp = int(time.time()) # Ensure time is imported

                # Pick a default voice or make it configurable if needed later
                chosen_voice_id = CASEY_VOICE_ID # Or FINN_VOICE_ID, or a narrator voice
                chosen_model_id = CASEY_TTS_MODEL_ID # Or your preferred model for this voice

                segment_filename_base = f"podcast_full_conversation_{request_timestamp}"

                print(f"Converting conversation to speech using voice {chosen_voice_id}...")
                generated_audio_filename_only = await text_to_speech_file(
                    text=conversation_text_for_tts, 
                    output_filename_base=segment_filename_base, # Pass base name
                    output_dir=audio_output_dir,
                    voice_id=chosen_voice_id,
                    model_id=chosen_model_id
                )

                if generated_audio_filename_only:
                    print(f"Audio file generated: {generated_audio_filename_only}")
                    final_state_dict["generated_audio_file"] = generated_audio_filename_only
                else:
                    print("Failed to generate audio file.")
                    final_state_dict["generated_audio_file"] = None 
            else:
                print("No conversation history to convert to speech.")
                final_state_dict["generated_audio_file"] = None
        else:
            print("Audio generation skipped based on request toggle.")
            final_state_dict["generated_audio_file"] = None # Ensure it's None if not generated
        # --- End Conditional TTS Generation ---

        return final_state_dict

    except Exception as e:
        print(f"Error during LangGraph or TTS execution: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error in podcast flow: {str(e)}")
    
@app.post("/submit_doubt/{thread_id}", response_model=AgentState)
async def submit_doubt_endpoint(thread_id: str, request: SubmitDoubtRequest):
    if not app_graph:
        raise HTTPException(status_code=500, detail="LangGraph app not initialized.")

    request_timestamp = int(time.time()) # For unique filenames if audio is generated
    print(f"Received doubt for thread_id: {thread_id}. Doubt: '{request.user_doubt_text}', Generate Audio: {request.generate_audio}")
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 15}
    
    input_update = {"user_doubt": request.user_doubt_text}

    try:
        final_state_dict = await app_graph.ainvoke(input_update, config=config)
        print(f"Graph resumed after doubt for {thread_id}. Turns: {final_state_dict.get('current_turn')}")

        if request.generate_audio:
            # Identify new parts of history to TTS (User Doubt + Finn's answer to doubt)
            history = final_state_dict.get("conversation_history", [])
            new_segments_for_tts = []
            if len(history) >= 2: # Heuristic: last two are doubt and answer
                if history[-2].startswith("User Doubt:") and history[-1].startswith("Factual Finn (addressing doubt):"):
                    new_segments_for_tts.append(history[-2]) # Optionally voice the user's doubt
                    new_segments_for_tts.append(history[-1]) # Finn's answer to doubt
            
            if new_segments_for_tts:
                audio_filename = await _generate_podcast_audio(
                    new_segments_for_tts, # Only new segments
                    request_timestamp,
                    "doubt_response"
                )
                final_state_dict["generated_audio_file"] = audio_filename
            else: # No new segments clearly identified or history too short
                final_state_dict["generated_audio_file"] = None
        else:
            final_state_dict["generated_audio_file"] = None
            
        return final_state_dict
    except Exception as e:
        print(f"Error processing doubt for thread_id {thread_id}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing doubt: {str(e)}")

@app.get("/get_podcast_audio/{filename}")
async def get_podcast_audio(filename: str):
    file_path = os.path.join(AUDIO_OUTPUT_DIR, filename) 
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='audio/mpeg', filename=filename)
    else:
        print(f"Audio file not found at: {file_path}")
        raise HTTPException(status_code=404, detail="Audio file not found.")
    
@app.post("/generate_themes", response_model=GuidingThemesOutput) # Respond with the Pydantic model
async def generate_themes_endpoint(request: GenerateThemesRequest):
    if not curious_casey_llm: # Using Casey's LLM for theme generation
        raise HTTPException(status_code=503, detail="Curious Casey LLM not initialized for theme generation.")

    print(f"Received request to generate themes for topic: {request.topic_info[:50]}...")

    # 1. Set up the PydanticOutputParser
    parser = PydanticOutputParser(pydantic_object=GuidingThemesOutput)

    # 2. Create a prompt template that includes the parser's format instructions
    # The original_prompt is what you passed from graph_orchestrator.py
    # We just need the topic_info here to build the new prompt for theme generation.

    theme_prompt_template_str = """Based on the following main topic:
    {topic_info}

    Generate a list of 3 to 4 key open-ended questions or themes that a podcast host (Curious Casey) should explore to cover this topic comprehensively for an audience. These themes will guide Curious Casey's questioning.

    {format_instructions}
    """

    prompt = PromptTemplate(
        template=theme_prompt_template_str,
        input_variables=["topic_info"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | curious_casey_llm | parser # Create a simple chain

    try:
        # Invoke the chain with the topic info
        # The parser will attempt to parse the LLM's string output into GuidingThemesOutput
        response_pydantic = await chain.ainvoke({"topic_info": request.topic_info})

        print(f"Generated and parsed themes: {response_pydantic.themes}")
        return response_pydantic # FastAPI will serialize this Pydantic model

    except Exception as e:
        # This can include OutputParserException if the LLM output doesn't match the schema
        print(f"Error in /generate_themes endpoint (LLM call or parsing): {e}\n{traceback.format_exc()}")
        # Fallback or raise error
        # For now, let's try to return a default if parsing fails, or raise an HTTP error
        # A more robust fallback might be needed if the LLM consistently fails to format.
        # For now, let's let it raise an HTTP error if parsing fails.
        raise HTTPException(status_code=500, detail=f"Error generating or parsing themes: {str(e)}")