from typing import List, Annotated, Optional
from typing_extensions import TypedDict
import operator
import httpx
import os
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# If you run this file directly for testing, uncomment and ensure .env is in this directory or project root
# from dotenv import load_dotenv
# load_dotenv()

class AgentState(TypedDict):
    initial_user_input: str
    casey_question: str
    finn_explanation: str
    conversation_history: Annotated[List[str], operator.add]
    current_turn: int
    generated_audio_file: Optional[str]
    user_doubt: Optional[str] # Field to store user's doubt

BACKEND_API_BASE_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000")

async def invoke_curious_casey(state: AgentState) -> dict:
    print("---NODE: CURIOUS_CASEY---")
    current_turn = state.get("current_turn", 0)
    
    context_for_casey = "" # This will be the statement Casey formulates a question about

    if current_turn == 0: # First turn for Casey in a regular flow
        context_for_casey = state.get("initial_user_input")
        if not context_for_casey:
            print("Error: Initial user input not found for Casey's first turn.")
            return {
                "casey_question": "Error: No initial topic provided for discussion.", 
                "conversation_history": ["Error: No initial topic for Casey."]
            }
        print(f"Casey (Turn {current_turn}) using initial input as context: '{context_for_casey[:70]}...'")
    else: # Subsequent turns for Casey, following Finn's last explanation
        context_for_casey = state.get("finn_explanation")
        if not context_for_casey:
            print("Error: Factual Finn's last explanation not found for Casey's follow-up.")
            return {
                "casey_question": "Error: Missing context from Factual Finn to formulate a question.", 
                "conversation_history": ["Error: Missing Finn's context for Casey."]
            }
        print(f"Casey (Turn {current_turn}) using Finn's last explanation as context: '{context_for_casey[:70]}...'")

    # The /ask_follow_up endpoint expects "previous_statement"
    payload = {"previous_statement": context_for_casey}
    api_url = f"{BACKEND_API_BASE_URL}/ask_follow_up"
    
    casey_question_text = "I'm not quite sure what to ask next." # Default/fallback
    new_history_entries = []

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(api_url, json=payload, timeout=180.0)
            response.raise_for_status()
            response_data = response.json()
            casey_question_text = response_data.get("follow_up_question", casey_question_text)
        
        print(f"Casey generated question: '{casey_question_text}'")
        if current_turn == 0 and state.get("initial_user_input"): # Add initial topic only once
            new_history_entries.append(f"Initial Topic: {state.get('initial_user_input')}")
        new_history_entries.append(f"Curious Casey: {casey_question_text}")

    except Exception as e:
        print(f"Error in invoke_curious_casey API call: {e}")
        new_history_entries.append(f"Error: Curious Casey failed to generate a question - {str(e)}")
    
    return {
        "casey_question": casey_question_text,
        "conversation_history": new_history_entries,
        "current_turn": current_turn + 1 # Increment turn count for regular AI-AI flow
    }

async def invoke_factual_finn_explain(state: AgentState) -> dict:
    print("---NODE: FACTUAL_FINN_EXPLAIN (Regular Explanation)---")
    casey_question = state.get("casey_question")
    initial_input = state.get("initial_user_input") # Finn always explains based on the initial input, guided by Casey

    if not casey_question or not initial_input:
        print("Error: Casey's question or initial input context not found for Factual Finn.")
        return {"finn_explanation": "Error: Missing context for explanation.", "conversation_history": ["Error: Missing context for Finn's explanation."]}

    print(f"Finn received question: '{casey_question}' regarding initial topic: '{initial_input[:70]}...'")
    
    finn_instruction = f"In response to the question: '{casey_question}', explain the relevant aspects of the following information: '{initial_input}' in a direct and factual style, focusing on key points."
    
    payload = {"information_text": initial_input, "instruction": finn_instruction}
    api_url = f"{BACKEND_API_BASE_URL}/explain"
    finn_explanation_text = "I could not provide an explanation at this time." # Default
    new_history_entry = ""

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(api_url, json=payload, timeout=180.0)
            response.raise_for_status()
            response_data = response.json()
            finn_explanation_text = response_data.get("explanation", finn_explanation_text)
        print(f"Finn generated explanation: '{finn_explanation_text[:100]}...'")
        new_history_entry = f"Factual Finn: {finn_explanation_text}"
    except Exception as e:
        print(f"Error in invoke_factual_finn_explain API call: {e}")
        new_history_entry = f"Error: Factual Finn failed to provide explanation - {str(e)}"
        
    return {
        "finn_explanation": finn_explanation_text,
        "conversation_history": [new_history_entry]
    }

async def invoke_factual_finn_on_doubt(state: AgentState) -> dict:
    print("---NODE: FACTUAL_FINN_ANSWER_DOUBT---")
    user_doubt = state.get("user_doubt")
    
    # Context for the doubt will be the conversation so far.
    # We can also include the initial topic for broader context if needed.
    conversation_so_far = "\n".join(state.get("conversation_history", []))
    initial_topic = state.get("initial_user_input", "the main topic")

    if not user_doubt:
        print("Warning: FACTUAL_FINN_ANSWER_DOUBT called without a user_doubt in state. Clearing doubt and proceeding.")
        return {"user_doubt": None, "finn_explanation": ""} 

    print(f"Finn received user doubt: '{user_doubt}'")

    # Instruction for Finn to address the doubt
    finn_instruction = (
        f"The ongoing podcast conversation is as follows:\n'''\n{conversation_so_far}\n'''\n\n"
        f"A user (audience member) has interrupted with the following doubt or question: '{user_doubt}'.\n"
        f"Please address this doubt directly and factually, using your knowledge of the overall topic: '{initial_topic}' and the preceding conversation as context. Focus on key points."
    )
    # The information_text for the /explain endpoint can be the initial topic or recent history
    # Let's pass the initial topic as the primary context, the instruction provides the doubt.
    payload = {
        "information_text": initial_topic, 
        "instruction": finn_instruction
    }
    api_url = f"{BACKEND_API_BASE_URL}/explain"
    finn_response_to_doubt = "I'm unable to address that specific doubt at the moment." # Default
    new_history_entries = [f"User Doubt: {user_doubt}"]

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(api_url, json=payload, timeout=180.0)
            response.raise_for_status()
            response_data = response.json()
            finn_response_to_doubt = response_data.get("explanation", finn_response_to_doubt)
        print(f"Finn generated response to doubt: '{finn_response_to_doubt[:100]}...'")
        new_history_entries.append(f"Factual Finn (addressing doubt): {finn_response_to_doubt}")
    except Exception as e:
        print(f"Error in invoke_factual_finn_on_doubt API call: {e}")
        new_history_entries.append(f"Error: Factual Finn failed to answer doubt - {str(e)}")
    
    return {
        "finn_explanation": finn_response_to_doubt, # This is Finn's answer to the doubt
        "conversation_history": new_history_entries,
        "user_doubt": None,  # Clear the doubt after addressing it
    }

MAX_CONVERSATION_TURNS = 3

def route_after_finn_node(state: AgentState) -> str:
    print(f"---ROUTER: route_after_finn_node (Current Turn: {state.get('current_turn', 0)}, User Doubt Present: {state.get('user_doubt') is not None})---")

    # Priority 1: Is there a user doubt to handle?
    if state.get("user_doubt"): 
        print("User doubt is present. Routing to FACTUAL_FINN_ANSWER_DOUBT.")
        return "FACTUAL_FINN_ANSWER_DOUBT"

    # Priority 2: If no doubt, check normal AI-AI turn limits.
    if state.get("current_turn", 0) >= MAX_CONVERSATION_TURNS:
        print("Max AI-AI turns reached (and no pending doubt). Ending conversation.")
        return END
    else:
        print("No pending doubt, continuing AI-AI conversation to Casey.")
        return "CURIOUS_CASEY"

# --- Define the LangGraph Workflow ---
print("Defining LangGraph workflow with HITL path...")
workflow = StateGraph(AgentState)

# Add the nodes
workflow.add_node("CURIOUS_CASEY", invoke_curious_casey)
workflow.add_node("FACTUAL_FINN_EXPLAIN", invoke_factual_finn_explain) # Node for regular explanations
workflow.add_node("FACTUAL_FINN_ANSWER_DOUBT", invoke_factual_finn_on_doubt) # Node for answering doubts

# Define the entry point and main flow
workflow.set_entry_point("CURIOUS_CASEY")
workflow.add_edge("CURIOUS_CASEY", "FACTUAL_FINN_EXPLAIN")

# Conditional routing after FACTUAL_FINN_EXPLAIN
# This node will be where an interruption might occur. If a doubt is submitted,
# the /submit_doubt endpoint will call ainvoke with user_doubt in the state.
# A more advanced graph could have a dedicated "check_for_doubt" node here.
# For now, let's assume if a doubt is present, the next invocation of the graph will handle it.
# The 'route_after_finn_node' will decide to continue to Casey or end.
workflow.add_conditional_edges(
    "FACTUAL_FINN_EXPLAIN",
    route_after_finn_node, 
    {
        "FACTUAL_FINN_ANSWER_DOUBT": "FACTUAL_FINN_ANSWER_DOUBT",
        "CURIOUS_CASEY": "CURIOUS_CASEY",
        END: END
    }
)

workflow.add_conditional_edges(
    "FACTUAL_FINN_ANSWER_DOUBT",
    route_after_finn_node, # After answering doubt, this router checks turns
    {
        "FACTUAL_FINN_ANSWER_DOUBT": "FACTUAL_FINN_ANSWER_DOUBT", # Should not be hit if doubt is cleared by the node
        "CURIOUS_CASEY": "CURIOUS_CASEY",
        END: END
    }
)

# Compile the graph
memory_saver = MemorySaver()
app_graph = workflow.compile(checkpointer=memory_saver)
print("LangGraph workflow compiled successfully with HITL routing and checkpointer.")