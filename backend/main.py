from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

OLLAMA_MODEL_NAME = "factual-finn" # The name you gave your model in Ollama (ollama create factual-finn ...)
OLLAMA_BASE_URL = "http://localhost:11434" # Default Ollama API URL

# Alpaca prompt template (same as used for fine tuning)
# We'll format this before sending it to Ollama
alpaca_prompt_template_str = """Below is an instruction that describes a task, paired with an input tht provides further context. Write a response that appropriately completes the request. 

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""

# Global variable for the langchain ChatOllama instance
chat_model = None

# --- FastAPI Lifespan for Ollama Client Initialization --- 
@asynccontextmanager
async def lifespan(app: FastAPI):
    global chat_model
    print(f"Initializing ChatOllama for model: {OLLAMA_MODEL_NAME} at {OLLAMA_BASE_URL}...")
    try:
        chat_model = ChatOllama(
            model=OLLAMA_MODEL_NAME, 
            base_url=OLLAMA_BASE_URL,
            temperature=0, # For factual, deterministic output
        )
        print("ChatOllama initialized. (Note: This doesn't confirm Ollama server is running or model is valid yet)")
    except Exception as e:
        print(f"Error initializing ChatOllama: {e}")
        chat_model = None

    yield # Application runs here
    
    print("FastAPI app shutting down.")

app = FastAPI(lifespan=lifespan)

# --- Pydantic Model for Request Body --- 
class ExplanationRequest(BaseModel):
    information_text: str
    instruction: str = "Explain the following information in a direct and factual style, focusing on key points."

# --- API Endpoints --- 
@app.get("/")
async def root():
    # Simple check if chat_model was attempted to be initialized
    if chat_model:
        return {"message": "AI Podcast Backend is running! Ollama client initialized."}
    else:
        return {"message": "AI Podcast Backend is running! OLLAMA CLIENT FAILED TO INITIALIZE."}

@app.post("/explain")
async def get_explanation(request: ExplanationRequest):
    if not chat_model:
        raise HTTPException(status_code=503, detail="Ollama service not initialized or unavailable.")

    print(f"Received request for explanation: {request.information_text[:50]}...") # Log first 50 chars

    # Format the full prompt using the Alpaca structure
    formatted_prompt = alpaca_prompt_template_str.format(
        instruction=request.instruction,
        input_text=request.information_text
    )
    
    try:
        print(f"Sending prompt to Ollama model '{OLLAMA_MODEL_NAME}':\n{formatted_prompt}")
        
        # Using LangChain to invoke the Ollama model
        response = await chat_model.ainvoke(formatted_prompt) # Use ainvoke for async
        
        # The response from ChatOllama.invoke is usually an AIMessage object.
        # The actual generated text is in its `content` attribute.
        explanation_text = response.content
        
        print(f"Received response from Ollama: {explanation_text[:100]}...") # Log first 100 chars of response

        # The GGUF model, especially Llama 3 based ones with proper templates,
        # should ideally stop generation correctly.
        # The `### Response:` part is part of our input prompt to guide the model.
        # The model's output should *only* be what Factual Finn would say.
        # No need for the complex string splitting we had before if Ollama + GGUF handles it well.
        return {"explanation": explanation_text.strip()}

    except Exception as e:
        print(f"Error during Ollama API call: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get explanation from Ollama: {str(e)}")
