from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
import uuid
import time
import traceback

from typing import Optional, List
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers.retry import RetryWithErrorOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from graph_orchestrator import app_graph, AgentState
from dotenv import load_dotenv

from fastapi.responses import FileResponse
from tts_utils import text_to_speech_elevenlabs_async as text_to_speech_file
from pydub import AudioSegment

from fastapi import File, UploadFile, Form
from langchain_community.document_loaders import PyPDFLoader
from rag_utils import create_vector_store_from_text, get_retriever_for_doc
from fastapi.concurrency import run_in_threadpool
import tempfile


load_dotenv()

# --- Configuration ---
OLLAMA_FACTUAL_FINN_MODEL_NAME = "factual-finn"
OLLAMA_CURIOUS_CASEY_MODEL_NAME = "curious-casey"
OLLAMA_GENERAL_MODEL_NAME = os.getenv("OLLAMA_GENERAL_MODEL", "llama3.1") # New: General purpose model
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# (Voice ID and Audio Dir constants remain the same)
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

# --- Global LLM clients ---
factual_finn_llm = None
curious_casey_llm = None
general_llm = None # New: General purpose client

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

    # <<<< NEW: Initialize General Purpose LLM >>>>
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
class StartPodcastRequest(BaseModel):
    initial_information: str
    doc_id: str
    generate_audio: bool = False
    user_provided_themes: Optional[List[str]] = None
class SubmitDoubtRequest(BaseModel):
    user_doubt_text: str
    generate_audio: bool = False
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


async def _generate_podcast_audio(conversation_history: list, request_timestamp: int, filename_prefix: str) -> Optional[str]:
    # (This function is complete and does not need changes)
    generated_audio_segments_paths = []; final_podcast_filename_only = None
    if not conversation_history: return None
    for i, turn_text in enumerate(conversation_history):
        speaker_label, text_to_speak, current_voice_id, current_model_id = "", "", "", ""
        if turn_text.startswith("Curious Casey:"): speaker_label, text_to_speak, current_voice_id, current_model_id = "Casey", turn_text.replace("Curious Casey: ", "", 1).strip(), CASEY_VOICE_ID, CASEY_TTS_MODEL_ID
        elif turn_text.startswith("Factual Finn (addressing doubt):"): speaker_label, text_to_speak, current_voice_id, current_model_id = "Finn_Doubt", turn_text.replace("Factual Finn (addressing doubt): ", "", 1).strip(), FINN_VOICE_ID, FINN_TTS_MODEL_ID
        elif turn_text.startswith("Factual Finn:"): speaker_label, text_to_speak, current_voice_id, current_model_id = "Finn", turn_text.replace("Factual Finn: ", "", 1).strip(), FINN_VOICE_ID, FINN_TTS_MODEL_ID
        elif turn_text.startswith("Initial Topic:"): speaker_label, text_to_speak, current_voice_id, current_model_id = "Narrator", turn_text.replace("Initial Topic: ", "", 1).strip(), NARRATOR_VOICE_ID, NARRATOR_TTS_MODEL_ID
        elif turn_text.startswith("User Doubt:"): continue
        else: continue
        if not text_to_speak: continue
        segment_filename_base = f"segment_{request_timestamp}_{i}_{speaker_label}"
        segment_filename = await text_to_speech_file(text=text_to_speak, output_filename_base=segment_filename_base, output_dir=AUDIO_OUTPUT_DIR, voice_id=current_voice_id, model_id=current_model_id)
        if segment_filename: generated_audio_segments_paths.append(os.path.join(AUDIO_OUTPUT_DIR, segment_filename))
    if generated_audio_segments_paths:
        combined_audio = AudioSegment.empty()
        for segment_path in generated_audio_segments_paths:
            if os.path.exists(segment_path):
                try: combined_audio += AudioSegment.from_mp3(segment_path)
                except Exception as e: print(f"Error loading segment {segment_path}: {e}")
            else: print(f"Audio segment file not found, skipping: {segment_path}")
        if len(combined_audio) > 0:
            final_podcast_filename_only = f"{filename_prefix}_{request_timestamp}.mp3"
            final_podcast_full_path = os.path.join(AUDIO_OUTPUT_DIR, final_podcast_filename_only)
            try: combined_audio.export(final_podcast_full_path, format="mp3"); return final_podcast_filename_only
            except Exception as e: print(f"Error exporting combined audio: {e}\n{traceback.format_exc()}")
    return None

# --- API Endpoints ---

@app.get("/")
async def root():
    finn_status = "Ready" if factual_finn_llm else "Failed"
    casey_status = "Ready" if curious_casey_llm else "Failed"
    general_status = "Ready" if general_llm else "Failed" # Added general status
    graph_status = "Ready" if app_graph else "Failed"
    return {
        "message": "AI Podcast Backend is running!",
        "factual_finn_status": finn_status,
        "curious_casey_status": casey_status,
        "general_llm_status": general_status,
        "langgraph_status": graph_status
    }

@app.post("/process_document_for_rag")
async def process_document_for_rag_endpoint(doc_id: str = Form(...), file: UploadFile = File(...)):
    # ... (This endpoint remains the same)
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
    if success: return {"message": f"Document '{file.filename}' processed for doc_id: {doc_id}"}
    else: raise HTTPException(status_code=500, detail=f"Failed to create vector store for doc_id: {doc_id}")

@app.post("/explain", response_model=dict)
async def get_explanation_from_finn_model(request: ExplanationRequest):
    # (Reverting retriever call to the robust version, simplifying prompt)
    if not factual_finn_llm: raise HTTPException(status_code=503, detail="Factual Finn (Ollama) service not initialized.")
    retriever = get_retriever_for_doc(request.doc_id)
    if not retriever: raise HTTPException(status_code=404, detail=f"No document found for doc_id: {request.doc_id}.")
    query_for_retrieval = request.instruction
    try:
        relevant_docs = await run_in_threadpool(retriever.get_relevant_documents, query_for_retrieval)
    except Exception as e:
        print(f"Error during RAG retrieval: {e}\n{traceback.format_exc()}"); raise HTTPException(status_code=500, detail="Failed to retrieve relevant documents.")
    retrieved_context = "\n\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else "No specific context found in the document for this query."
    final_instruction_for_finn = f"Please answer the following question in a direct and factual style, focusing on key points. Use only the provided context.\n\nQuestion: {request.instruction}"
    final_input_text_for_finn = f"Context to use for the answer:\n'''\n{retrieved_context}\n'''"
    prompt_to_ollama = alpaca_prompt_template_str.format(instruction=final_instruction_for_finn, input_text=final_input_text_for_finn)
    try:
        response = await factual_finn_llm.ainvoke(prompt_to_ollama)
        return {"explanation": response.content.strip()}
    except Exception as e: print(f"Error in /explain endpoint (Ollama call): {e}\n{traceback.format_exc()}"); raise HTTPException(status_code=500, detail=f"Error generating RAG explanation: {str(e)}")

@app.post("/ask_follow_up", response_model=dict)
async def get_question_from_casey_model(request: FollowUpRequest): # This endpoint remains the same
    # ... your working /ask_follow_up endpoint code ...
    if not curious_casey_llm: raise HTTPException(status_code=503, detail="Curious Casey (Ollama) service not initialized.")
    try:
        casey_direct_input = f"The knowledgeable AI just said: '{request.previous_statement}'"
        response = await curious_casey_llm.ainvoke(casey_direct_input)
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

        # The parser returns a Pydantic *object*, not a dict. Let's name it appropriately.
        parsed_output = await retry_parser.aparse_with_prompt(response.content, prompt_value)
        
        # Access the 'themes' attribute directly on the object.
        print(f"Generated and parsed themes: {parsed_output.themes}")
        
        # Return the Pydantic object. FastAPI will automatically convert it to JSON.
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

        # This returns a CoverageEvaluationOutput object
        evaluation_result = await retry_parser.aparse_with_prompt(response.content, prompt_value)
        
        print(f"Coverage evaluation result: {evaluation_result}")
        
        # We return the object, and FastAPI serializes it based on the response_model
        return evaluation_result 
    except Exception as e:
        print(f"Error during coverage evaluation LLM call or parsing: {e}\n{traceback.format_exc()}")
        # This fallback needs to return a valid object matching the Pydantic model
        return CoverageEvaluationOutput(
            is_covered=False, 
            reasoning=f"Failed to evaluate due to an error: {str(e)}"
        )

@app.post("/initiate_podcast_flow", response_model=AgentState)
async def initiate_podcast_flow_endpoint(request: StartPodcastRequest):
    if not app_graph: raise HTTPException(status_code=500, detail="LangGraph application not initialized.")
    conversation_thread_id = str(uuid.uuid4()); request_timestamp = int(time.time()); print(f"Initiating podcast flow. Thread ID: {conversation_thread_id}, Generate Audio: {request.generate_audio}")
    if request.user_provided_themes: print(f"User provided themes: {request.user_provided_themes}")
    initial_graph_input = AgentState(initial_user_input=request.initial_information, doc_id=request.doc_id, casey_question="", finn_explanation="", conversation_history=[], current_turn=0, generated_audio_file=None, user_doubt=None, guiding_themes=request.user_provided_themes, covered_themes=[], current_targeted_theme=None)
    config = {"configurable": {"thread_id": conversation_thread_id}, "recursion_limit": 15}
    try:
        final_state_dict = await app_graph.ainvoke(initial_graph_input, config=config)
        print(f"LangGraph AI-AI flow complete for {conversation_thread_id}. Turns: {final_state_dict.get('current_turn')}")
        if request.generate_audio: final_state_dict["generated_audio_file"] = await _generate_podcast_audio(final_state_dict.get("conversation_history", []), request_timestamp, "podcast_initiate")
        else: final_state_dict["generated_audio_file"] = None
        return final_state_dict
    except Exception as e: print(f"Error in /initiate_podcast_flow endpoint: {e}\n{traceback.format_exc()}"); raise HTTPException(status_code=500, detail=f"Error in podcast flow execution: {str(e)}")
    
@app.post("/submit_doubt/{thread_id}", response_model=AgentState)
async def submit_doubt_endpoint(thread_id: str, request: SubmitDoubtRequest):
    if not app_graph: raise HTTPException(status_code=500, detail="LangGraph app not initialized.")
    request_timestamp = int(time.time()); print(f"Received doubt for thread_id: {thread_id}. Doubt: '{request.user_doubt_text}', Generate Audio: {request.generate_audio}")
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 15}
    input_update = {"user_doubt": request.user_doubt_text}
    try:
        final_state_dict = await app_graph.ainvoke(input_update, config=config)
        print(f"Graph resumed after doubt for {thread_id}. Turns: {final_state_dict.get('current_turn')}")
        if request.generate_audio:
            history = final_state_dict.get("conversation_history", []); new_segments_for_tts = []
            if len(history) >= 2 and history[-2].startswith("User Doubt:") and history[-1].startswith("Factual Finn (addressing doubt):"): new_segments_for_tts.extend(history[-2:])
            if new_segments_for_tts: final_state_dict["generated_audio_file"] = await _generate_podcast_audio(new_segments_for_tts, request_timestamp, "doubt_response")
            else: final_state_dict["generated_audio_file"] = None
        else: final_state_dict["generated_audio_file"] = None
        return final_state_dict
    except Exception as e: print(f"Error processing doubt for thread_id {thread_id}: {e}\n{traceback.format_exc()}"); raise HTTPException(status_code=500, detail=f"Error processing doubt: {str(e)}")
    
@app.get("/get_podcast_audio/{filename}")
async def get_podcast_audio(filename: str):
    file_path = os.path.join(AUDIO_OUTPUT_DIR, filename); 
    if os.path.exists(file_path): return FileResponse(file_path, media_type='audio/mpeg', filename=filename)
    else: raise HTTPException(status_code=404, detail="Audio file not found.")

