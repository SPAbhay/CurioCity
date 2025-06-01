from typing import List, Annotated, Optional
from typing_extensions import TypedDict
import operator
import httpx
import os
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import traceback

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
    user_doubt: Optional[str] 
    guiding_themes: Optional[List[str]]
    covered_themes: Optional[List[str]]

BACKEND_API_BASE_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000")

async def invoke_curious_casey(state: AgentState) -> dict:
    print("---NODE: CURIOUS_CASEY---")
    current_turn = state.get("current_turn", 0)
    guiding_themes = state.get("guiding_themes", [])
    covered_themes = state.get("covered_themes", [])
    
    context_for_casey_prompt = "" # This will be the statement Casey formulates a question about

    if current_turn == 0: # First turn for Casey in a regular flow
        context_for_casey_prompt = state.get("initial_user_input")
        if not context_for_casey_prompt:
            print("Error: Initial user input not found for Casey's first turn.")
            return {
                "casey_question": "Error: No initial topic provided for discussion.", 
                "conversation_history": ["Error: No initial topic for Casey."]
            }
        print(f"Casey (Turn {current_turn}) using initial input as context: '{context_for_casey_prompt[:70]}...'")
    else: # Subsequent turns for Casey, following Finn's last explanation
        context_for_casey_prompt = state.get("finn_explanation")
        if not context_for_casey_prompt:
            print("Error: Factual Finn's last explanation not found for Casey's follow-up.")
            return {
                "casey_question": "Error: Missing context from Factual Finn to formulate a question.", 
                "conversation_history": ["Error: Missing Finn's context for Casey."]
            }
        print(f"Casey (Turn {current_turn}) using Finn's last explanation as context: '{context_for_casey_prompt[:70]}...'")

    next_theme_to_ask_about = None
    if guiding_themes:
        for theme in guiding_themes:
            if theme not in covered_themes:
                next_theme_to_ask_about = theme
                break
    
    # Construct the prompt for Casey's Ollama model
    # This prompt needs to instruct Casey to ask a question related to 'next_theme_to_ask_about'
    # while also considering 'context_for_casey_prompt' (Finn's last statement or initial input).
    # This is where Casey's fine-tuning for asking follow-ups is leveraged.
    # The "previous_statement" for the /ask_follow_up API will be a combination.

    if next_theme_to_ask_about:
        # We're telling Casey what theme to focus on, based on Finn's last statement
        api_payload_statement = (
            f"Regarding the topic '{next_theme_to_ask_about}', "
            f"and considering what was just said: '{context_for_casey_prompt}'"
        )
        print(f"Casey focusing on theme: '{next_theme_to_ask_about}'")
    else:
        # All themes covered, or no themes provided. Ask a general follow-up.
        api_payload_statement = context_for_casey_prompt
        print("Casey asking a general follow-up as themes are covered or absent.")

    
    payload = {"previous_statement": api_payload_statement}
    api_url = f"{BACKEND_API_BASE_URL}/ask_follow_up"
    casey_question_text = "What else can you tell me about this?" # Default
    new_history_entries = []
    updated_covered_themes = list(covered_themes) # Make a copy to modify


    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(api_url, json=payload, timeout=180.0)
            response.raise_for_status()
            response_data = response.json()
            casey_question_text = response_data.get("follow_up_question", casey_question_text)
        
        print(f"Casey generated question: '{casey_question_text}'")
        if current_turn == 0 and state.get("initial_user_input"):
            new_history_entries.append(f"Initial Topic: {state.get('initial_user_input')}")
        new_history_entries.append(f"Curious Casey: {casey_question_text}")

        # Optimistically mark the theme as covered if we asked about it
        if next_theme_to_ask_about and next_theme_to_ask_about not in updated_covered_themes:
            updated_covered_themes.append(next_theme_to_ask_about)
            print(f"Theme '{next_theme_to_ask_about}' marked as covered.")

    except Exception as e:
        print(f"Error in invoke_curious_casey API call: {e}")
        new_history_entries.append(f"Error: Curious Casey failed to generate a question - {str(e)}")
    
    return {
        "casey_question": casey_question_text,
        "conversation_history": new_history_entries,
        "current_turn": current_turn + 1,
        "covered_themes": updated_covered_themes 
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
    
async def generate_guiding_themes(state: AgentState) -> dict:
    print("---NODE: GENERATE_GUIDING_THEMES---")

    if state.get("guiding_themes") is not None and isinstance(state.get("guiding_themes"), list):
        user_themes = state.get("guiding_themes")
        if user_themes and all(isinstance(theme, str) for theme in user_themes):
            print(f"Using user-provided guiding themes: {user_themes}")
            return {"guiding_themes": user_themes, "covered_themes": state.get("covered_themes", [])}
        else:
            print("User-provided themes were empty or not valid. Proceeding to generate themes.")

    initial_input = state.get("initial_user_input")
    if not initial_input:
        print("Error: Initial user input not found for generating themes.")
        return {"guiding_themes": [], "covered_themes": []}

    api_url = f"{BACKEND_API_BASE_URL}/generate_themes"
    generated_themes_list = []
    default_themes_on_failure = [ # Define default themes once
        f"Explore the core definition and key features of '{initial_input[:30]}...'.",
        f"Discuss the primary applications or consequences related to '{initial_input[:30]}...'.",
        "Consider any interesting examples or future outlook."
    ]

    try:
        async with httpx.AsyncClient() as client:
            # The /generate_themes endpoint now just needs the topic_info
            # The specific prompt for theme generation is handled within that endpoint
            response = await client.post(api_url, json={"topic_info": initial_input}, timeout=120.0)
            response.raise_for_status()
            response_data = response.json() # Expects {"themes": ["theme1", "theme2", ...]}

            # The response should now be a list of strings directly from the Pydantic model
            generated_themes_list = response_data.get("themes", []) 

            if not generated_themes_list:
                print("API returned no themes or 'themes' key missing. Using default themes.")
                generated_themes_list = default_themes_on_failure
            else:
                print(f"AI Generated and Parsed Guiding Themes: {generated_themes_list}")

    except Exception as e:
        print(f"Error calling /generate_themes API or processing its response: {e}\n{traceback.format_exc()}")
        generated_themes_list = default_themes_on_failure
        print(f"Using default guiding themes due to API error: {generated_themes_list}")

    return {"guiding_themes": generated_themes_list, "covered_themes": []}

MAX_CONVERSATION_TURNS = 3

def route_after_finn_node(state: AgentState) -> str:
    print(f"---ROUTER: route_after_finn_node (Current Turn: {state.get('current_turn', 0)}, User Doubt: {state.get('user_doubt') is not None})---")
    
    if state.get("user_doubt"):
        print("User doubt is present. Routing to FACTUAL_FINN_ANSWER_DOUBT.")
        return "FACTUAL_FINN_ANSWER_DOUBT"
    
    # Check if all guiding themes are covered
    guiding_themes = state.get("guiding_themes")
    covered_themes = state.get("covered_themes")
    if guiding_themes and covered_themes is not None and len(covered_themes) >= len(guiding_themes):
        print("All guiding themes covered. Ending conversation.")
        return END

    if state.get("current_turn", 0) >= MAX_CONVERSATION_TURNS:
        print("Max AI-AI turns reached (and no pending doubt or all themes not covered). Ending conversation.")
        return END
    else:
        print("No pending doubt, themes remaining or turns left. Continuing to Casey.")
        return "CURIOUS_CASEY"
    
async def route_initial_or_doubt(state: AgentState) -> str:
    print("---ROUTER: INITIAL_OR_DOUBT_ROUTER (Condition Function)---")
    if state.get("user_doubt"):
        print("User doubt detected at entry/resume. Routing to FACTUAL_FINN_ANSWER_DOUBT.")
        return "FACTUAL_FINN_ANSWER_DOUBT"
    else:
        print("No initial user doubt. Routing to GENERATE_GUIDING_THEMES first.")
        return "GENERATE_GUIDING_THEMES" 


# --- Define the LangGraph Workflow ---
print("Defining LangGraph workflow with conditional entry point and guiding themes...")
workflow = StateGraph(AgentState)

workflow.add_node("CURIOUS_CASEY", invoke_curious_casey)
workflow.add_node("FACTUAL_FINN_EXPLAIN", invoke_factual_finn_explain)
workflow.add_node("FACTUAL_FINN_ANSWER_DOUBT", invoke_factual_finn_on_doubt)
workflow.add_node("GENERATE_GUIDING_THEMES", generate_guiding_themes)

# Set a conditional entry point
workflow.set_conditional_entry_point(
    route_initial_or_doubt, # This function's string output determines the first actual node
    {
        "GENERATE_GUIDING_THEMES": "GENERATE_GUIDING_THEMES",
        "FACTUAL_FINN_ANSWER_DOUBT": "FACTUAL_FINN_ANSWER_DOUBT",
        
    }
)


# Existing edges for the main conversation flow
workflow.add_edge("GENERATE_GUIDING_THEMES", "CURIOUS_CASEY") 
workflow.add_edge("CURIOUS_CASEY", "FACTUAL_FINN_EXPLAIN")

# Existing conditional routing after FACTUAL_FINN_EXPLAIN
workflow.add_conditional_edges(
    "FACTUAL_FINN_EXPLAIN",
    route_after_finn_node, 
    {
        "FACTUAL_FINN_ANSWER_DOUBT": "FACTUAL_FINN_ANSWER_DOUBT",
        "CURIOUS_CASEY": "CURIOUS_CASEY",
        END: END
    }
)

# Existing conditional routing after FACTUAL_FINN_ANSWER_DOUBT
workflow.add_conditional_edges(
    "FACTUAL_FINN_ANSWER_DOUBT",
    route_after_finn_node, 
    {
        "CURIOUS_CASEY": "CURIOUS_CASEY",
        END: END
    }
)

# Compile the graph
memory_saver = MemorySaver()
app_graph = workflow.compile(checkpointer=memory_saver)
print("LangGraph workflow compiled successfully with conditional entry point and HITL.")