from typing import List, Annotated, Optional, Dict
from typing_extensions import TypedDict
import operator
import httpx
import os
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import traceback
import random

class AgentState(TypedDict):
    initial_user_input: str
    casey_question: str
    finn_explanation: str
    conversation_history: Annotated[List[str], operator.add]
    current_turn: int
    final_podcast_file: Optional[str]
    user_doubt: Optional[str] 
    doc_id: Optional[str]
    guiding_themes: Optional[List[str]]
    covered_themes: Optional[List[str]]
    current_targeted_theme: Optional[str] 
    generation_mode: str 
    turns_on_current_theme: int
    target_turns_for_theme: int 
    audio_log: Dict[int, str]
    persona_name: str
    generate_audio: bool 
    current_sources: List[str]

BACKEND_API_BASE_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000")
TURNS_PER_THEME_RANGE = (1, 3) 

# --- Node Functions ---
# Note: None of the node functions need to be changed. They will continue to
# append clean text to `conversation_history` as before.

async def invoke_curious_casey(state: AgentState) -> dict:
    print("---NODE: CURIOUS_CASEY---")
    current_turn = state.get("current_turn", 0)
    guiding_themes = state.get("guiding_themes", [])
    covered_themes = state.get("covered_themes", [])
    current_targeted_theme = state.get("current_targeted_theme")
    turns_on_current_theme = state.get("turns_on_current_theme", 0)
    target_turns_for_theme = state.get("target_turns_for_theme", 0)
    
    context_for_casey_prompt = state.get("finn_explanation") if current_turn > 0 else state.get("initial_user_input")
    if not context_for_casey_prompt:
        return {"casey_question": "Error: Missing context.", "current_targeted_theme": current_targeted_theme, "turns_on_current_theme": turns_on_current_theme, "target_turns_for_theme": target_turns_for_theme}
    
    if not current_targeted_theme:
        next_theme_to_ask_about = next((theme for theme in guiding_themes if theme not in covered_themes), None)
        if next_theme_to_ask_about:
            target_turns_for_theme = random.randint(*TURNS_PER_THEME_RANGE)
            print(f"Casey starting NEW theme: '{next_theme_to_ask_about}' (Targeting {target_turns_for_theme} turns)")
            api_payload_statement = (
                f"Let's start a new topic. Regarding the guiding theme '{next_theme_to_ask_about}', "
                f"and considering the overall subject of '{state.get('initial_user_input')}', please ask an engaging opening question."
            )
            current_targeted_theme = next_theme_to_ask_about
            turns_on_current_theme = 0
        else:
            print("Casey asking a general wrap-up question.")
            api_payload_statement = f"Considering all we've discussed, including '{context_for_casey_prompt}', let's try to summarize the key takeaway."
    
    else:
        print(f"Casey asking FOLLOW-UP on theme: '{current_targeted_theme}' (Turn {turns_on_current_theme + 1}/{target_turns_for_theme})")
        api_payload_statement = (
            f"Regarding the current theme '{current_targeted_theme}', "
            f"and based on what was just said: '{context_for_casey_prompt}', please ask a relevant follow-up question."
        )

    payload = {"previous_statement": api_payload_statement}
    api_url = f"{BACKEND_API_BASE_URL}/ask_follow_up"
    casey_question_text = ""
    new_history_entries = []

    for attempt in range(3):
        print(f"Attempt {attempt + 1} to get question from Casey...")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(api_url, json=payload, timeout=180.0)
                response.raise_for_status()
                casey_question_text = response.json().get("follow_up_question", "").strip()
            if casey_question_text:
                print(f"Casey generated question: '{casey_question_text}'")
                break
            else:
                print("Casey returned an empty question. Retrying...")
        except Exception as e:
            print(f"Error in invoke_curious_casey API call on attempt {attempt + 1}: {e}")
    
    if not casey_question_text:
        print("Casey failed to generate a question after multiple attempts.")
        casey_question_text = "Could you please elaborate on the last point?"
        new_history_entries.append("Error: Casey failed to generate a question, using a fallback.")

    if current_turn == 0:
        new_history_entries.append(f"Initial Topic: {state.get('initial_user_input')}")
    new_history_entries.append(f"Curious Casey: {casey_question_text}")
    
    return {
        "casey_question": casey_question_text,
        "conversation_history": new_history_entries,
        "current_turn": current_turn + 1,
        "current_targeted_theme": current_targeted_theme,
        "turns_on_current_theme": turns_on_current_theme,
        "target_turns_for_theme": target_turns_for_theme
    }


async def invoke_knowledgeable_persona_explain(state: AgentState) -> dict:
    print("---NODE: KNOWLEDGEABLE_PERSONA_EXPLAIN (RAG Enabled)---")
    casey_question = state.get("casey_question")
    doc_id = state.get("doc_id")
    initial_topic = state.get("initial_user_input")
    persona_name = state.get("persona_name", "factual-finn") 

    if not all([casey_question, doc_id]): 
        return {"finn_explanation": "Error: Missing context/doc_id for RAG.", "conversation_history": ["System: Error: Missing context for Finn."]}

    print(f"Persona '{persona_name}' received question: '{casey_question}' for doc_id: '{doc_id}'")
    
    payload = {
        "instruction": casey_question, 
        "doc_id": doc_id, 
        "original_topic_or_context": initial_topic,
        "persona_name": persona_name
    }
    api_url = f"{BACKEND_API_BASE_URL}/explain"
    explanation_text, new_history_entry = f"I could not provide an explanation as {persona_name} at this time.", ""
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(api_url, json=payload, timeout=180.0); response.raise_for_status()
            response_data = response.json()
            explanation_text = response_data.get("explanation", explanation_text)
            # NEW: Get the sources from the response
            sources = response_data.get("sources", [])
        
        persona_display_name = persona_name.replace('-', ' ').title()
        print(f"{persona_display_name} (RAG) generated explanation: '{explanation_text[:100]}...'")
        new_history_entry = f"{persona_display_name}: {explanation_text}"

    except Exception as e:
        print(f"Error in invoke_knowledgeable_persona_explain (RAG) API call: {e}"); new_history_entry = f"Error: Persona failed to provide RAG explanation - {str(e)}"
        sources = []
    
    return {"finn_explanation": explanation_text, "conversation_history": [new_history_entry], "current_sources": sources}

async def evaluate_theme_coverage(state: AgentState) -> dict:
    print("---NODE: EVALUATE_THEME_COVERAGE---")
    current_targeted_theme = state.get("current_targeted_theme")
    covered_themes = state.get("covered_themes", [])
    turns_on_current_theme = state.get("turns_on_current_theme", 0) + 1
    target_turns_for_theme = state.get("target_turns_for_theme", 1) # Default to 1 to be safe

    updated_covered_themes = list(covered_themes)
    next_targeted_theme = current_targeted_theme

    if current_targeted_theme and turns_on_current_theme >= target_turns_for_theme:
        print(f"Theme '{current_targeted_theme}' has been discussed for {turns_on_current_theme}/{target_turns_for_theme} turns and is now marked as covered.")
        if current_targeted_theme not in updated_covered_themes:
            updated_covered_themes.append(current_targeted_theme)
        
        next_targeted_theme = None
        turns_on_current_theme = 0
    elif current_targeted_theme:
        print(f"Theme '{current_targeted_theme}' has been discussed for {turns_on_current_theme}/{target_turns_for_theme} turns. Continuing.")

    return {
        "covered_themes": updated_covered_themes,
        "current_targeted_theme": next_targeted_theme,
        "turns_on_current_theme": turns_on_current_theme,
        "target_turns_for_theme": target_turns_for_theme
    }


async def invoke_factual_finn_on_doubt(state: AgentState) -> dict:
    print("---NODE: FACTUAL_FINN_ANSWER_DOUBT---")
    user_doubt, conversation_so_far, initial_topic = state.get("user_doubt"), "\n".join(state.get("conversation_history", [])), state.get("initial_user_input", "the main topic")
    doc_id = state.get("doc_id")
    if not user_doubt or not doc_id: return {"user_doubt": None, "finn_explanation": "Error: Missing doubt or doc_id to answer."}
    print(f"Finn received user doubt: '{user_doubt}'")
    finn_instruction = (
        f"The ongoing podcast conversation is as follows:\n'''\n{conversation_so_far}\n'''\n\n"
        f"A user (audience member) has interrupted with the following doubt or question: '{user_doubt}'.\n"
        f"Please address this doubt directly and factually, using your knowledge of the document about '{initial_topic}' and the preceding conversation as context. Focus on key points."
    )
    payload = {"instruction": finn_instruction, "doc_id": doc_id, "original_topic_or_context": initial_topic}
    api_url = f"{BACKEND_API_BASE_URL}/explain"
    finn_response_to_doubt, new_history_entries = "I'm unable to address that specific doubt at the moment.", [f"User Doubt: {user_doubt}"]
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(api_url, json=payload, timeout=180.0); response.raise_for_status()
            finn_response_to_doubt = response.json().get("explanation", finn_response_to_doubt)
        print(f"Finn generated response to doubt: '{finn_response_to_doubt[:100]}...'")
        new_history_entries.append(f"Factual Finn (addressing doubt): {finn_response_to_doubt}")
    except Exception as e:
        print(f"Error in invoke_factual_finn_on_doubt API call: {e}"); new_history_entries.append(f"Error: Factual Finn failed to answer doubt - {str(e)}")
    return {"finn_explanation": finn_response_to_doubt, "conversation_history": new_history_entries, "user_doubt": None}


def should_interrupt(state: AgentState) -> bool:
    """Decide whether to interrupt the graph based on the generation mode."""
    return state.get("generation_mode") == "interactive"


def route_after_evaluation(state: AgentState) -> str:
    print(f"---ROUTER: (Mode: {state.get('generation_mode')}, Covered: {len(state.get('covered_themes',[]))}/{len(state.get('guiding_themes',[]))}, Doubt: {state.get('user_doubt') is not None})---")
    
    if state.get("user_doubt"):
        print("User doubt detected. Routing to FACTUAL_FINN_ANSWER_DOUBT.")
        return "FACTUAL_FINN_ANSWER_DOUBT"
    
    guiding_themes = state.get("guiding_themes", [])
    covered_themes = state.get("covered_themes", [])

    if len(covered_themes) >= len(guiding_themes):
        print("All guiding themes have been evaluated as covered. Ending conversation.")
        return END
    
    print("Themes remaining. Continuing to Casey.")
    return "CURIOUS_CASEY"


# --- Define the LangGraph Workflow ---
print("Defining LangGraph workflow with EXPLICIT THEME INPUT...")
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("CURIOUS_CASEY", invoke_curious_casey)
workflow.add_node("KNOWLEDGEABLE_PERSONA_EXPLAIN", invoke_knowledgeable_persona_explain)
workflow.add_node("EVALUATE_THEME_COVERAGE", evaluate_theme_coverage)
workflow.add_node("FACTUAL_FINN_ANSWER_DOUBT", invoke_factual_finn_on_doubt)

workflow.set_entry_point("CURIOUS_CASEY")

# Define edges
workflow.add_edge("CURIOUS_CASEY", "KNOWLEDGEABLE_PERSONA_EXPLAIN")
workflow.add_edge("KNOWLEDGEABLE_PERSONA_EXPLAIN", "EVALUATE_THEME_COVERAGE")

workflow.add_conditional_edges(
    "EVALUATE_THEME_COVERAGE",
    route_after_evaluation,
    {
        "CURIOUS_CASEY": "CURIOUS_CASEY",
        "FACTUAL_FINN_ANSWER_DOUBT": "FACTUAL_FINN_ANSWER_DOUBT",
        END: END
    }
)

workflow.add_conditional_edges(
    "FACTUAL_FINN_ANSWER_DOUBT",
    route_after_evaluation,
    {
        "CURIOUS_CASEY": "CURIOUS_CASEY",
        END: END
    }
)

memory_saver = MemorySaver()
app_graph = workflow.compile(
    checkpointer=memory_saver,
    interrupt_after=["EVALUATE_THEME_COVERAGE"],
)
print("LangGraph workflow compiled with INTERRUPT trigger for human-in-the-loop.")