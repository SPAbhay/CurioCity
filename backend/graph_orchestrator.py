from typing import List, Annotated, Optional
from typing_extensions import TypedDict
import operator
import httpx
import os
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import traceback

class AgentState(TypedDict):
    initial_user_input: str
    casey_question: str
    finn_explanation: str
    conversation_history: Annotated[List[str], operator.add]
    current_turn: int
    generated_audio_file: Optional[str]
    user_doubt: Optional[str] 
    doc_id: Optional[str]
    guiding_themes: Optional[List[str]]
    covered_themes: Optional[List[str]]
    current_targeted_theme: Optional[str] # New: To track the theme Casey is currently asking about

BACKEND_API_BASE_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000")

# --- Node Functions ---
async def invoke_curious_casey(state: AgentState) -> dict:
    # This function is modified to set 'current_targeted_theme'
    print("---NODE: CURIOUS_CASEY---")
    current_turn = state.get("current_turn", 0)
    guiding_themes = state.get("guiding_themes", [])
    covered_themes = state.get("covered_themes", [])
    
    context_for_casey_prompt = state.get("finn_explanation") if current_turn > 0 else state.get("initial_user_input")
    if not context_for_casey_prompt:
        return {"casey_question": "Error: Missing context.", "current_targeted_theme": None}
    
    print(f"Casey (Turn {current_turn}) using context: '{context_for_casey_prompt[:70]}...'")
    next_theme_to_ask_about = next((theme for theme in guiding_themes if theme not in covered_themes), None)
    
    api_payload_statement = context_for_casey_prompt
    if next_theme_to_ask_about:
        api_payload_statement = (
            f"Regarding the guiding theme '{next_theme_to_ask_about}', "
            f"and considering what was just said: '{context_for_casey_prompt}'"
        )
        print(f"Casey focusing on theme: '{next_theme_to_ask_about}'")
    else:
        print("Casey asking a general follow-up as all themes are now covered.")
        # If all themes are covered, the router should have already ended the conversation.
        # This part might serve as a fallback. Let's make the question more conclusive.
        api_payload_statement = f"Considering all we've discussed, including '{context_for_casey_prompt}', let's try to summarize the key takeaway."

    payload = {"previous_statement": api_payload_statement}
    api_url = f"{BACKEND_API_BASE_URL}/ask_follow_up"
    casey_question_text = "Is there anything else to add on this topic?"
    new_history_entries = []

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(api_url, json=payload, timeout=180.0)
            response.raise_for_status()
            casey_question_text = response.json().get("follow_up_question", casey_question_text)
        
        print(f"Casey generated question: '{casey_question_text}'")
        if current_turn == 0:
            new_history_entries.append(f"Initial Topic: {state.get('initial_user_input')}")
        new_history_entries.append(f"Curious Casey: {casey_question_text}")
    except Exception as e:
        print(f"Error in invoke_curious_casey API call: {e}"); new_history_entries.append(f"Error: Casey failed to ask question - {str(e)}")
    
    return {
        "casey_question": casey_question_text,
        "conversation_history": new_history_entries,
        "current_turn": current_turn + 1,
        "current_targeted_theme": next_theme_to_ask_about # Set the theme being targeted
    }


async def invoke_factual_finn_explain(state: AgentState) -> dict:
    # This function remains the same
    print("---NODE: FACTUAL_FINN_EXPLAIN (RAG Enabled)---")
    casey_question, doc_id, initial_topic = state.get("casey_question"), state.get("doc_id"), state.get("initial_user_input")
    if not all([casey_question, doc_id]): return {"finn_explanation": "Error: Missing context/doc_id for RAG.", "conversation_history": ["Error: Missing context for Finn."]}
    print(f"Finn received question: '{casey_question}' for doc_id: '{doc_id}'")
    payload = {"instruction": casey_question, "doc_id": doc_id, "original_topic_or_context": initial_topic}
    api_url = f"{BACKEND_API_BASE_URL}/explain"
    finn_explanation_text, new_history_entry = "I could not provide an explanation at this time using RAG.", ""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(api_url, json=payload, timeout=180.0); response.raise_for_status()
            finn_explanation_text = response.json().get("explanation", finn_explanation_text)
        print(f"Finn (RAG) generated explanation: '{finn_explanation_text[:100]}...'")
        new_history_entry = f"Factual Finn: {finn_explanation_text}"
    except Exception as e:
        print(f"Error in invoke_factual_finn_explain (RAG) API call: {e}"); new_history_entry = f"Error: Factual Finn failed to provide RAG explanation - {str(e)}"
    return {"finn_explanation": finn_explanation_text, "conversation_history": [new_history_entry]}


# NEW NODE FUNCTION
async def evaluate_theme_coverage(state: AgentState) -> dict:
    print("---NODE: EVALUATE_THEME_COVERAGE---")
    targeted_theme = state.get("current_targeted_theme")
    casey_question = state.get("casey_question")
    finn_explanation = state.get("finn_explanation")
    covered_themes = state.get("covered_themes", [])
    
    # If there was no theme being targeted in this turn, no need to evaluate.
    if not targeted_theme:
        print("No specific theme was targeted. Skipping coverage evaluation.")
        return {"current_targeted_theme": None}

    print(f"Evaluating if Finn's answer for '{casey_question[:50]}...' covered the theme: '{targeted_theme}'")
    payload = {"targeted_theme": targeted_theme, "question": casey_question, "answer": finn_explanation}
    api_url = f"{BACKEND_API_BASE_URL}/evaluate_coverage"
    updated_covered_themes = list(covered_themes)

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(api_url, json=payload, timeout=60.0)
            response.raise_for_status()
            is_covered = response.json().get("is_covered", False)
            
            if is_covered:
                print(f"Evaluation result: Theme '{targeted_theme}' IS covered.")
                if targeted_theme not in updated_covered_themes:
                    updated_covered_themes.append(targeted_theme)
            else:
                print(f"Evaluation result: Theme '{targeted_theme}' IS NOT yet fully covered. It will be re-queued.")
    except Exception as e:
        print(f"Error during theme coverage evaluation API call: {e}. Assuming theme not covered.")

    return {
        "covered_themes": updated_covered_themes,
        "current_targeted_theme": None # Clear the targeted theme after evaluation for the next turn
    }


async def invoke_factual_finn_on_doubt(state: AgentState) -> dict:
    # This function remains the same
    print("---NODE: FACTUAL_FINN_ANSWER_DOUBT---")
    # ... your existing logic for this function ...
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


MAX_CONVERSATION_TURNS = 3

def route_after_evaluation(state: AgentState) -> str:
    print(f"---ROUTER: route_after_evaluation (Turn: {state.get('current_turn', 0)}, User Doubt: {state.get('user_doubt') is not None})---")

    # If a doubt was just answered, we continue the main flow.
    # The 'user_doubt' field is cleared after being answered.
    # This router is now the main controller for the conversation flow.
    if state.get("user_doubt"):
        print("User doubt detected. Routing to FACTUAL_FINN_ANSWER_DOUBT.")
        return "FACTUAL_FINN_ANSWER_DOUBT"

    guiding_themes = state.get("guiding_themes")
    covered_themes = state.get("covered_themes")

    if guiding_themes and covered_themes is not None and len(covered_themes) >= len(guiding_themes):
        print("All guiding themes have been evaluated as covered. Ending conversation.")
        return END

    if state.get("current_turn", 0) >= MAX_CONVERSATION_TURNS:
        print("Max AI-AI turns reached. Ending conversation.")
        return END

    print("Themes remaining or no doubt present. Continuing to Casey.")
    return "CURIOUS_CASEY"


# --- Define the LangGraph Workflow ---
print("Defining LangGraph workflow with EXPLICIT THEME INPUT...")
workflow = StateGraph(AgentState)

# Add nodes
# Note: "GENERATE_GUIDING_THEMES" node has been removed.
workflow.add_node("CURIOUS_CASEY", invoke_curious_casey)
workflow.add_node("FACTUAL_FINN_EXPLAIN", invoke_factual_finn_explain)
workflow.add_node("EVALUATE_THEME_COVERAGE", evaluate_theme_coverage)
workflow.add_node("FACTUAL_FINN_ANSWER_DOUBT", invoke_factual_finn_on_doubt)

# The entry point is now always CURIOUS_CASEY.
# The graph assumes themes are pre-populated in the state.
workflow.set_entry_point("CURIOUS_CASEY")

# Define edges
workflow.add_edge("CURIOUS_CASEY", "FACTUAL_FINN_EXPLAIN")
workflow.add_edge("FACTUAL_FINN_EXPLAIN", "EVALUATE_THEME_COVERAGE")

# Conditional routing AFTER theme coverage has been evaluated
workflow.add_conditional_edges(
    "EVALUATE_THEME_COVERAGE",
    route_after_evaluation,
    {
        "CURIOUS_CASEY": "CURIOUS_CASEY",
        "FACTUAL_FINN_ANSWER_DOUBT": "FACTUAL_FINN_ANSWER_DOUBT",
        END: END
    }
)

# Conditional routing AFTER a doubt is answered
workflow.add_conditional_edges(
    "FACTUAL_FINN_ANSWER_DOUBT",
    route_after_evaluation,
    {
        "CURIOUS_CASEY": "CURIOUS_CASEY",
        END: END
    }
)


# Compile the graph
memory_saver = MemorySaver()
app_graph = workflow.compile(
    checkpointer=memory_saver,
    interrupt_after=["EVALUATE_THEME_COVERAGE"],
)
print("LangGraph workflow compiled with INTERRUPT trigger for human-in-the-loop.")
