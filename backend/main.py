from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from graph_orchestrator import app_graph, AgentState

from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
OLLAMA_FACTUAL_FINN_MODEL_NAME = "factual-finn"
OLLAMA_CURIOUS_CASEY_MODEL_NAME = "curious-casey" # New model name for Casey
OLLAMA_BASE_URL = "http://localhost:11434"

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
    
@app.post("/initiate_podcast_flow", response_model=AgentState) # Respond with the final state
async def initiate_podcast_flow(request: StartPodcastRequest):
    if not app_graph: # Should not happen if graph_orchestrator.py is imported correctly
        raise HTTPException(status_code=500, detail="LangGraph application not initialized.")
    if not factual_finn_llm or not curious_casey_llm: # Check if Ollama clients are ready
         raise HTTPException(status_code=503, detail="One or more Ollama LLM clients are not ready.")


    print(f"Received request to initiate podcast flow with: {request.initial_information[:100]}...")

    # Prepare the initial state for the graph
    initial_graph_input = AgentState(
        initial_user_input=request.initial_information,
        casey_question="",        # Will be filled by CURIOUS_CASEY node
        finn_explanation="",      # Will be filled by FACTUAL_FINN node
        conversation_history=[],  # Will be populated by the nodes
        current_turn=0            # Initialize current_turn
    )

    try:
        print("Invoking LangGraph app_graph...")
        # Use ainvoke to get the final state after the graph completes
        # For more detailed streaming of events (like individual node outputs as they happen),
        # you would use app_graph.astream_events(...) and process the events.
        # For now, let's get the final state.
        final_state = await app_graph.ainvoke(initial_graph_input, {"recursion_limit": 10}) # Added recursion_limit

        # The final_state will be an AgentState dictionary
        print(f"LangGraph execution complete. Final state: {final_state}")
        return final_state # FastAPI will serialize this TypedDict to JSON

    except Exception as e:
        print(f"Error during LangGraph execution: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error in podcast flow: {str(e)}")