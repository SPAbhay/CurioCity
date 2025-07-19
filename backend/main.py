from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
import uuid
import time
import traceback
import re

from typing import Optional, List, Dict
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers.retry import RetryWithErrorOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from graph_orchestrator import app_graph, AgentState
from dotenv import load_dotenv

from fastapi.responses import FileResponse
from tts_utils import text_to_speech_elevenlabs_async
from pydub import AudioSegment

from fastapi import File, UploadFile, Form
from langchain_community.document_loaders import PyPDFLoader
from rag_utils import create_vector_store_from_text, get_retriever_for_doc
from fastapi.concurrency import run_in_threadpool
import tempfile


load_dotenv()

# --- Configuration (remains the same) ---
OLLAMA_FACTUAL_FINN_MODEL_NAME = "factual-finn"
OLLAMA_CURIOUS_CASEY_MODEL_NAME = "curious-casey"
OLLAMA_GENERAL_MODEL_NAME = os.getenv("OLLAMA_GENERAL_MODEL", "llama3.1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

CASEY_VOICE_ID = os.getenv("CASEY_VOICE_ID", "F2OOWcJMWhX8wCsyT0oR")
FINN_VOICE_ID = os.getenv("FINN_VOICE_ID", "IFEvkitzF8OoHeggkJUu")
NARRATOR_VOICE_ID = os.getenv("NARRATOR_VOICE_ID", CASEY_VOICE_ID)
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

# --- Global LLM clients (remains the same) ---
factual_finn_llm = None
curious_casey_llm = None
general_llm = None

# --- Lifespan function (remains the same) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global factual_finn_llm, curious_casey_llm, general_llm, app_graph
    os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
    
    # Initialize Factual Finn
    print(f"Initializing ChatOllama for Factual Finn: {OLLAMA_FACTUAL_FINN_MODEL_NAME}...")
    try:
        factual_finn_llm = ChatOllama(model=OLLAMA_FACTUAL_FINN_MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0)
        print("Factual Finn ChatOllama client initialized.")
    except Exception as e:
        print(f"Error initializing Factual Finn ChatOllama client: {e}\n{traceback.format_exc()}")
    
    # Initialize Curious Casey
    print(f"Initializing ChatOllama for Curious Casey: {OLLAMA_CURIOUS_CASEY_MODEL_NAME}...")
    try:
        curious_casey_llm = ChatOllama(model=OLLAMA_CURIOUS_CASEY_MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0.7)
        print("Curious Casey ChatOllama client initialized.")
    except Exception as e:
        print(f"Error initializing Curious Casey ChatOllama client: {e}\n{traceback.format_exc()}")

    # Initialize General Purpose LLM
    print(f"Initializing ChatOllama for General Use: {OLLAMA_GENERAL_MODEL_NAME}...")
    try:
        general_llm = ChatOllama(model=OLLAMA_GENERAL_MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0)
        print("General Purpose ChatOllama client initialized.")
    except Exception as e:
        print(f"Error initializing General Purpose ChatOllama client: {e}\n{traceback.format_exc()}")

    if app_graph:
        print("LangGraph app_graph loaded from orchestrator.")
    else:
        print("CRITICAL Error: LangGraph app_graph not loaded! Check graph_orchestrator.py.")
    yield
    print("FastAPI app shutting down.")


app = FastAPI(lifespan=lifespan)
origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# --- Pydantic Models (remain the same) ---
class ProposeThemesResponse(BaseModel):
    thread_id: str
    themes: List[str]

class ProposeThemesRequest(BaseModel):
    user_provided_themes: Optional[List[str]] = None
    doc_id: str

class StartPodcastRequest(BaseModel):
    final_themes: List[str]
    doc_id: str
    generate_audio: bool = False
    generation_mode: str

class PodcastFlowResponse(BaseModel):
    thread_id: str
    state: AgentState

class SubmitDoubtRequest(BaseModel):
    user_doubt_text: str

class CreateFullPodcastResponse(BaseModel):
    final_podcast_file: Optional[str]

class ExplanationRequest(BaseModel):
    instruction: str
    doc_id: str
    original_topic_or_context: Optional[str] = None

class FollowUpRequest(BaseModel):
    previous_statement: str

class GenerateThemesRequest(BaseModel):
    topic_info: str

class GuidingThemesOutput(BaseModel):
    themes: List[str] = Field(description="A list of 3 to 4 key open-ended questions or themes for the podcast.")

class CoverageEvaluationRequest(BaseModel):
    targeted_theme: str
    question: str
    answer: str

class CoverageEvaluationOutput(BaseModel):
    is_covered: bool = Field(description="Set to true if the theme was adequately covered, false otherwise.")
    reasoning: str = Field(description="A brief explanation for why the theme was or was not considered covered.")

class ProcessTextRequest(BaseModel):
    doc_id: str
    text_content: str

class DocumentProcessedResponse(BaseModel):
    message: str
    doc_id: str
    topic_summary: str

# --- Helper Functions ---

async def _summarize_text_for_topic(text_content: str) -> str:
    # This function remains the same
    if not general_llm:
        return "Summary not available."
    
    prompt = (
        "You are an expert at summarizing content. "
        "Based on the following excerpt from a document, please generate a concise, one-sentence topic summary suitable for a podcast title. "
        "Your response should contain ONLY the summary text and nothing else. Do not include any preamble.\n\n"
        f"EXCERPT:\n'''\n{text_content[:2000]}\n'''\n\n"
        "CONCISE TOPIC SUMMARY:"
    )
    try:
        response = await general_llm.ainvoke(prompt)
        
        raw_summary = response.content.strip()
        if ':' in raw_summary:
            summary = raw_summary.split(':', 1)[-1].strip()
        else:
            summary = raw_summary
        summary = summary.replace('"', '')
        
        print(f"Generated topic summary: {summary}")
        return summary
    except Exception as e:
        print(f"Error during topic summarization: {e}")
        return "Could not determine topic automatically."


async def _generate_audio_for_turn(turn_text: str, turn_index: int) -> Optional[str]:
    # This function is correct and remains the same
    speaker_label, text_to_speak, current_voice_id, current_model_id = "", "", "", ""

    if turn_text.startswith("Curious Casey:"):
        speaker_label, text_to_speak = "Casey", turn_text.replace("Curious Casey: ", "", 1).strip()
        current_voice_id, current_model_id = CASEY_VOICE_ID, CASEY_TTS_MODEL_ID
    elif turn_text.startswith("Factual Finn (addressing doubt):"):
        speaker_label, text_to_speak = "Finn_Doubt", turn_text.replace("Factual Finn (addressing doubt): ", "", 1).strip()
        current_voice_id, current_model_id = FINN_VOICE_ID, FINN_TTS_MODEL_ID
    elif turn_text.startswith("Factual Finn:"):
        speaker_label, text_to_speak = "Finn", turn_text.replace("Factual Finn: ", "", 1).strip()
        current_voice_id, current_model_id = FINN_VOICE_ID, FINN_TTS_MODEL_ID
    elif turn_text.startswith("Initial Topic:"):
        speaker_label, text_to_speak = "Narrator", turn_text.replace("Initial Topic: ", "", 1).strip()
        current_voice_id, current_model_id = NARRATOR_VOICE_ID, NARRATOR_TTS_MODEL_ID
    else:
        return None

    if not text_to_speak:
        return None

    segment_filename_base = f"segment_{int(time.time())}_{turn_index}_{speaker_label}"
    audio_filename = await text_to_speech_elevenlabs_async(
        text=text_to_speak,
        output_filename_base=segment_filename_base,
        output_dir=AUDIO_OUTPUT_DIR,
        voice_id=current_voice_id,
        model_id=current_model_id
    )
    return audio_filename


# REWRITTEN: This function now uses the new `audio_log` field to avoid the duplication bug.
async def _generate_audio_and_update_log(state_dict: dict, thread_id: str) -> dict:
    """
    Generates audio for any new turns and updates the `audio_log` in the state.
    This log is replaced on each update, which is the key to fixing the bug.
    """
    config = {"configurable": {"thread_id": thread_id}}
    history = state_dict.get("conversation_history", [])
    # Get the existing log or start a new one
    current_audio_log = state_dict.get("audio_log", {})
    
    log_was_updated = False
    # Use a copy to avoid modifying the dictionary while iterating
    new_audio_log = current_audio_log.copy()

    for i, turn in enumerate(history):
        # If we haven't already processed this turn number, generate audio
        if i not in new_audio_log:
            audio_filename = await _generate_audio_for_turn(turn, i)
            if audio_filename:
                print(f"Generated audio for turn {i}: {audio_filename}")
                new_audio_log[i] = audio_filename
                log_was_updated = True

    if log_was_updated:
        print("Updating state with new audio log.")
        # Update the state. Since `audio_log` has no `operator.add`, this is a clean replacement.
        app_graph.update_state(config, {"audio_log": new_audio_log})
        # Update the local dictionary that will be returned to the frontend
        state_dict["audio_log"] = new_audio_log

    return state_dict

# UPDATED: This function now uses the `audio_log` dictionary for combining.
def _combine_audio_files(audio_log: dict, output_dir: str, thread_id: str) -> Optional[str]:
    if not audio_log:
        return None

    # Sort the filenames by turn index (the dictionary key) to ensure correct order
    sorted_filenames = [audio_log[key] for key in sorted(audio_log.keys())]
    
    print(f"Combining {len(sorted_filenames)} audio segments for thread {thread_id}...")
    combined_audio = AudioSegment.empty()
    for filename in sorted_filenames:
        segment_path = os.path.join(output_dir, filename)
        if os.path.exists(segment_path):
            try:
                combined_audio += AudioSegment.from_mp3(segment_path)
            except Exception as e:
                print(f"Error loading segment {segment_path}: {e}")
        else:
            print(f"Audio segment file not found, skipping: {segment_path}")
    
    if len(combined_audio) > 0:
        final_podcast_filename_only = f"full_podcast_{thread_id}_{int(time.time())}.mp3"
        final_podcast_full_path = os.path.join(output_dir, final_podcast_filename_only)
        try:
            combined_audio.export(final_podcast_full_path, format="mp3")
            print(f"Exported combined audio to: {final_podcast_full_path}")
            return final_podcast_filename_only
        except Exception as e:
            print(f"Error exporting combined audio: {e}\n{traceback.format_exc()}")
    
    return None

# --- API Endpoints ---

@app.get("/")
async def root():
    # This endpoint remains the same
    finn_status = "Ready" if factual_finn_llm else "Failed"
    casey_status = "Ready" if curious_casey_llm else "Failed"
    general_status = "Ready" if general_llm else "Failed"
    graph_status = "Ready" if app_graph else "Failed"
    return {
        "message": "AI Podcast Backend is running!",
        "factual_finn_status": finn_status,
        "curious_casey_status": casey_status,
        "general_llm_status": general_status,
        "langgraph_status": graph_status
    }

@app.post("/process_text_for_rag", response_model=DocumentProcessedResponse)
async def process_text_for_rag_endpoint(request: ProcessTextRequest):
    # UPDATED: Initialize AgentState with the new audio_log field
    if not request.text_content.strip():
        raise HTTPException(status_code=400, detail="No text content provided.")
    
    success = await run_in_threadpool(create_vector_store_from_text, request.doc_id, request.text_content)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to create vector store for doc_id: {request.doc_id}")

    topic_summary = await _summarize_text_for_topic(request.text_content)
    
    config = {"configurable": {"thread_id": request.doc_id}}
    initial_graph_input = AgentState(
        initial_user_input=topic_summary,
        doc_id=request.doc_id,
        guiding_themes=[],
        covered_themes=[],
        casey_question="",
        finn_explanation="",
        conversation_history=[],
        current_turn=0,
        final_podcast_file=None,
        user_doubt=None,
        current_targeted_theme=None,
        generation_mode="interactive",
        turns_on_current_theme=0,
        target_turns_for_theme=0,
        audio_log={}
    )
    app_graph.update_state(config, initial_graph_input)

    return DocumentProcessedResponse(
        message=f"Text content processed for doc_id: {request.doc_id}",
        doc_id=request.doc_id,
        topic_summary=topic_summary
    )

@app.post("/process_document_for_rag", response_model=DocumentProcessedResponse)
async def process_document_for_rag_endpoint(doc_id: str = Form(...), file: UploadFile = File(...)):
    # UPDATED: Initialize AgentState with the new audio_log field
    if not file.filename.endswith(".pdf"): raise HTTPException(status_code=400, detail="Invalid file type.")
    text_content, tmp_file_path = "", ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file: content = await file.read(); tmp_file.write(content); tmp_file_path = tmp_file.name
        loader = PyPDFLoader(tmp_file_path); documents = await run_in_threadpool(loader.load); text_content = "\n\n".join([doc.page_content for doc in documents])
    except Exception as e: raise HTTPException(status_code=500, detail=f"Error processing PDF file: {str(e)}")
    finally:
        if os.path.exists(tmp_file_path): os.remove(tmp_file_path)
    if not text_content.strip(): raise HTTPException(status_code=400, detail="No text content extracted from PDF.")
    
    success = await run_in_threadpool(create_vector_store_from_text, doc_id, text_content)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to create vector store for doc_id: {doc_id}")

    topic_summary = await _summarize_text_for_topic(text_content)
    
    config = {"configurable": {"thread_id": doc_id}}
    initial_graph_input = AgentState(
        initial_user_input=topic_summary,
        doc_id=doc_id,
        guiding_themes=[],
        covered_themes=[],
        casey_question="",
        finn_explanation="",
        conversation_history=[],
        current_turn=0,
        final_podcast_file=None,
        user_doubt=None,
        current_targeted_theme=None,
        generation_mode="interactive",
        turns_on_current_theme=0,
        target_turns_for_theme=0,
        audio_log={}
    )
    app_graph.update_state(config, initial_graph_input)

    return DocumentProcessedResponse(
        message=f"Document '{file.filename}' processed for doc_id: {doc_id}",
        doc_id=doc_id,
        topic_summary=topic_summary
    )


# ... (explain, ask_follow_up, generate_themes, evaluate_coverage, propose_themes endpoints remain the same) ...
@app.post("/explain", response_model=dict)
async def get_explanation_from_finn_model(request: ExplanationRequest):
    if not factual_finn_llm: raise HTTPException(status_code=503, detail="Factual Finn (Ollama) service not initialized.")
    retriever = get_retriever_for_doc(request.doc_id)
    if not retriever: raise HTTPException(status_code=404, detail=f"No document found for doc_id: {request.doc_id}.")
    query_for_retrieval = request.instruction
    try:
        relevant_docs = await retriever.ainvoke(query_for_retrieval)
    except Exception as e:
        print(f"Error during RAG retrieval: {e}\n{traceback.format_exc()}"); raise HTTPException(status_code=500, detail="Failed to retrieve relevant documents.")
    retrieved_context = "\n\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else "No specific context found in the document for this query."
    final_instruction_for_finn = f"Please answer the following question in a direct and factual style, focusing on key points. Use only the provided context.\n\nQuestion: {request.instruction}"
    final_input_text_for_finn = f"Context to use for the answer:\n'''\n{retrieved_context}\n'''"
    prompt_to_ollama = alpaca_prompt_template_str.format(instruction=final_instruction_for_finn, input_text=final_input_text_for_finn)
    try:
        response = await factual_finn_llm.ainvoke(prompt_to_ollama, stop=["### Instruction:"])
        return {"explanation": response.content.strip()}
    except Exception as e: print(f"Error in /explain endpoint (Ollama call): {e}\n{traceback.format_exc()}"); raise HTTPException(status_code=500, detail=f"Error generating RAG explanation: {str(e)}")

@app.post("/ask_follow_up", response_model=dict)
async def get_question_from_casey_model(request: FollowUpRequest):
    if not curious_casey_llm: raise HTTPException(status_code=503, detail="Curious Casey (Ollama) service not initialized.")
    try:
        casey_direct_input = f"The knowledgeable AI just said: '{request.previous_statement}'"
        response = await curious_casey_llm.ainvoke(casey_direct_input, stop=["The knowledgeable AI just said:"])
        return {"follow_up_question": response.content.strip()}
    except Exception as e: print(f"Error in /ask_follow_up endpoint: {e}\n{traceback.format_exc()}"); raise HTTPException(status_code=500, detail=f"Error generating question: {str(e)}")

@app.post("/generate_themes", response_model=GuidingThemesOutput)
async def generate_themes_endpoint(request: GenerateThemesRequest):
    if not general_llm:
        raise HTTPException(status_code=503, detail="General purpose LLM not initialized for theme generation.")

    parser = PydanticOutputParser(pydantic_object=GuidingThemesOutput)
    retry_parser = RetryWithErrorOutputParser.from_llm(parser=parser, llm=general_llm)

    theme_prompt_template = PromptTemplate(
        template=(
            "You are an assistant that creates a structured plan for a podcast episode. "
            "Based on the main topic provided, generate a list of 3 to 4 key open-ended questions or themes that a podcast host should explore to cover the topic comprehensively.\n\n"
            "MAIN TOPIC:\n{topic_info}\n\n"
            "{format_instructions}"
        ),
        input_variables=["topic_info"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = theme_prompt_template | general_llm

    try:
        prompt_value = await theme_prompt_template.ainvoke({"topic_info": request.topic_info})
        response = await chain.ainvoke({"topic_info": request.topic_info})
        parsed_output = await retry_parser.aparse_with_prompt(response.content, prompt_value)
        print(f"Generated and parsed themes: {parsed_output.themes}")
        return parsed_output
    except Exception as e:
        print(f"Error in /generate_themes endpoint (LLM call or parsing): {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error generating or parsing themes: {str(e)}")
    
@app.post("/evaluate_coverage", response_model=CoverageEvaluationOutput)
async def evaluate_coverage_endpoint(request: CoverageEvaluationRequest):
    if not general_llm:
        raise HTTPException(status_code=503, detail="LLM for evaluation not initialized.")
    
    print(f"Received request to evaluate coverage for theme: '{request.targeted_theme}'")
    parser = PydanticOutputParser(pydantic_object=CoverageEvaluationOutput)
    retry_parser = RetryWithErrorOutputParser.from_llm(parser=parser, llm=general_llm)

    evaluation_prompt_template = PromptTemplate(
        template=(
            "You are a strict, logical evaluator. Your task is to determine if an expert's answer adequately covers a specific theme.\n"
            "Analyze the provided information and respond ONLY with a JSON object matching the schema.\n\n"
            "Guiding Theme: '{targeted_theme}'\n"
            "Question Asked: '{question}'\n"
            "Expert's Answer: '{answer}'\n\n"
            "Based *only* on the expert's answer, was the guiding theme adequately covered?\n\n"
            "{format_instructions}"
        ),
        input_variables=["targeted_theme", "question", "answer"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = evaluation_prompt_template | general_llm
    
    try:
        input_data = {
            "targeted_theme": request.targeted_theme,
            "question": request.question,
            "answer": request.answer
        }
        prompt_value = await evaluation_prompt_template.ainvoke(input_data)
        response = await chain.ainvoke(input_data)
        evaluation_result = await retry_parser.aparse_with_prompt(response.content, prompt_value)
        print(f"Coverage evaluation result: {evaluation_result}")
        return evaluation_result
    except Exception as e:
        print(f"Error during coverage evaluation LLM call or parsing: {e}\n{traceback.format_exc()}")
        return CoverageEvaluationOutput(
            is_covered=False,
            reasoning=f"Failed to evaluate due to an error: {str(e)}"
        )

@app.post("/propose_themes", response_model=ProposeThemesResponse)
async def propose_themes_endpoint(request: ProposeThemesRequest):
    thread_id = request.doc_id
    print(f"Proposing themes for thread_id: {thread_id}")
    
    config = {"configurable": {"thread_id": thread_id}}
    
    current_state_snapshot = app_graph.get_state(config)
    
    topic_summary = current_state_snapshot.values.get("initial_user_input", "No topic summary found.")
    
    themes = []
    if request.user_provided_themes:
        print(f"Using user-provided themes: {request.user_provided_themes}")
        themes = request.user_provided_themes
    else:
        print(f"No user themes provided, generating themes with AI based on topic: '{topic_summary}'")
        try:
            generated_themes_output = await generate_themes_endpoint(
                GenerateThemesRequest(topic_info=topic_summary)
            )
            themes = generated_themes_output.themes
        except Exception as e:
            print(f"Error generating themes: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate guiding themes.")
    
    app_graph.update_state(config, {"guiding_themes": themes})
    print(f"Themes saved for thread_id: {thread_id}")

    return ProposeThemesResponse(thread_id=thread_id, themes=themes)

@app.post("/initiate_podcast_flow/{thread_id}", response_model=AgentState)
async def initiate_podcast_flow_endpoint(thread_id: str, request: StartPodcastRequest):
    if not app_graph:
        raise HTTPException(status_code=500, detail="LangGraph application not initialized.")
    
    print(f"--- Initiating podcast flow for thread_id: {thread_id} (Mode: {request.generation_mode}) ---")
    config = {"configurable": {"thread_id": thread_id}}
    
    state_update = {
        "guiding_themes": request.final_themes,
        "generation_mode": request.generation_mode
    }
    app_graph.update_state(config, state_update)

    try:
        final_state_dict = await app_graph.ainvoke(None, config=config)
        
        if request.generation_mode == "bulk":
            print("Bulk mode detected. Running graph to completion...")
            while len(final_state_dict.get("covered_themes", [])) < len(final_state_dict.get("guiding_themes", [])):
                print(f"Bulk mode: Continuing... (Covered {len(final_state_dict.get('covered_themes', []))}/{len(final_state_dict.get('guiding_themes', []))})")
                final_state_dict = await app_graph.ainvoke(None, config=config)
                if final_state_dict.get("current_turn", 0) > 15:
                    print("Bulk mode safety break triggered after 15 turns.")
                    break
            print("Bulk mode run complete.")

        # UPDATED: Call the rewritten audio generation function
        final_state_dict = await _generate_audio_and_update_log(final_state_dict, thread_id)
        
        return final_state_dict
        
    except Exception as e:
        print(f"Error in /initiate_podcast_flow endpoint: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error in podcast flow execution: {str(e)}")
        
@app.post("/submit_doubt/{thread_id}", response_model=AgentState)
async def submit_doubt_endpoint(thread_id: str, request: SubmitDoubtRequest):
    if not app_graph: raise HTTPException(status_code=500, detail="LangGraph app not initialized.")
    print(f"Received doubt for thread_id: {thread_id}. Doubt: '{request.user_doubt_text}'")
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 15}
    input_update = {"user_doubt": request.user_doubt_text}
    try:
        final_state_dict = await app_graph.ainvoke(input_update, config=config)
        print(f"Graph resumed after doubt for {thread_id}. Turns: {final_state_dict.get('current_turn')}")

        # UPDATED: Call the rewritten audio generation function
        final_state_dict = await _generate_audio_and_update_log(final_state_dict, thread_id)
        
        return final_state_dict
    except Exception as e: print(f"Error processing doubt for thread_id {thread_id}: {e}\n{traceback.format_exc()}"); raise HTTPException(status_code=500, detail=f"Error processing doubt: {str(e)}")
    
@app.post("/continue_flow/{thread_id}", response_model=AgentState)
async def continue_podcast_flow_endpoint(thread_id: str):
    if not app_graph:
        raise HTTPException(status_code=500, detail="LangGraph app not initialized.")
    
    print(f"--- Resuming podcast flow for thread_id: {thread_id} ---")
    config = {"configurable": {"thread_id": thread_id}}

    try:
        final_state_dict = await app_graph.ainvoke(None, config=config)
        print(f"Graph resumed and paused again for {thread_id}. Current turns: {final_state_dict.get('current_turn')}")
        
        # UPDATED: Call the rewritten audio generation function
        final_state_dict = await _generate_audio_and_update_log(final_state_dict, thread_id)
        
        return final_state_dict

    except Exception as e:
        print(f"Error resuming flow for thread_id {thread_id}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error resuming podcast flow: {str(e)}")

# UPDATED: This endpoint now uses the audio_log.
@app.post("/create_full_podcast/{thread_id}", response_model=CreateFullPodcastResponse)
async def create_full_podcast_endpoint(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    current_state = app_graph.get_state(config)
    if not current_state:
        raise HTTPException(status_code=404, detail="Podcast session not found.")

    audio_log = current_state.values.get("audio_log", {})

    if not audio_log:
        raise HTTPException(status_code=400, detail="No audio segments found to combine.")

    final_filename = await run_in_threadpool(
        _combine_audio_files, audio_log, AUDIO_OUTPUT_DIR, thread_id
    )

    if not final_filename:
        raise HTTPException(status_code=500, detail="Failed to create combined podcast audio.")

    # Save the final filename to the state
    app_graph.update_state(config, {"final_podcast_file": final_filename})

    return CreateFullPodcastResponse(final_podcast_file=final_filename)

@app.get("/get_podcast_audio/{filename}")
async def get_podcast_audio(filename: str):
    # This endpoint remains the same and works for both segments and the final file
    file_path = os.path.join(AUDIO_OUTPUT_DIR, filename);
    if os.path.exists(file_path): return FileResponse(file_path, media_type='audio/mpeg', filename=filename)
    else: raise HTTPException(status_code=404, detail="Audio file not found.")