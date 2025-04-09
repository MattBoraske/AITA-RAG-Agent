from llama_index.core import PromptTemplate

class AITA_Prompt_Library:
    """AITA Prompt Library"""

    AITA_context = (
        "AITA (Am I The A**hole) is a format where people share personal conflicts and ask for judgment.\n"
        "In these situations:\n"
        "- The writer describes a specific conflict or dilemma they're involved in.\n"
        "- They explain their actions and the actions of other involved parties.\n"
        "- They share relevant context like relationships, history, and circumstances.\n"
        "- The core question is always about who's at fault in the conflict.\n"
        "- Judgments focus on actions and choices, not on judging people as individuals.\n"
    )

    AITA_classifications = [
        "you're the a**hole (YTA) when the writer is causing the conflict",
        "not the a**hole (NTA) when someone other than the writer is causing the conflict",
        "no a**holes here (NAH) when no one is causing the conflict",
        "everyone sucks here (ESH) when everyone is causing the conflict.",
    ]

    AITA_rules = (
        f"\t- Your AITA classification choices are limited to {', '.join(AITA_classifications)}\n"
        #f"\t- If the context has a YTA judgment, you MUST classify the current situation as YTA.\n"
        #f"\t- If the context has a NTA judgment, you MUST classify the current situation as NTA.\n"
        #f"\t- The same applies for NAH and ESH - your classification MUST mirror the context.\n"
        f"\t- You MUST maintain consistency with the reasoning in the provided context and are NOT allowed to make independent moral judgments.\n"
        f"\t- You are NOT allowed to cite the specific details from the context examples to support your judgement.\n"
        f"\t- Use the context ONLY as a template for:\n"
        f"\t\t- Writing in a similar style (phrasing and expressions) and tone (casual vs formal)\n"
        f"\t\t- Identifying the right amount of detail in justifications\n"
        f"\t\t- Presenting your verdict strictly in the form of the AITA classification followed by the justification\n"
        f"\t- Do NOT use racist, derogatory, or explicit language even if its used in the judgement example.\n"
    )

    PROMPTS = {
        'AITA_text_qa_template': PromptTemplate(
        f"{AITA_context}\n"
        "Provide your judgment of this AITA conflict:\n"
        "---------------------\n"
        "{query_str}\n"
        "---------------------\n\n"
        "You MUST follow these rules:\n"
        f"\t- You MUST judge the ORIGINAL AITA conflict and NOT the NEW conflict.\n"
        f"{AITA_rules}\n"
        "Your initial judgment: "
    ),
            
        'AITA_text_qa_RAG_template': PromptTemplate(
        f"{AITA_context}\n"
        "Study this NEW AITA conflict and judgement carefully:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n\n"
        "Using the above as context, provide your initial judgment of this ORIGINAL AITA conflict:\n"
        "---------------------\n"
        "{query_str}\n"
        "---------------------\n\n"
        "You MUST follow these rules:\n"
        f"\t- You MUST judge the ORIGINAL AITA conflict and NOT the NEW conflict.\n"
        f"{AITA_rules}\n"
        "Your initial judgment: "
    ),
    
    'AITA_refine_qa_RAG_template': PromptTemplate(
      f"{AITA_context}\n\n"
      "You are refining your judgment on this ORIGINAL AITA conflict:\n"
      "---------------------\n"      
      "{query_str}\n"
      "---------------------\n\n"
      "This is your previous judgment of the ORIGINAL AITA conflict:\n"
      "---------------------\n"   
      "{existing_answer}\n\n"
      "---------------------\n\n"
      "Now study this potentially SIMILAR AITA conflict and judgement:\n"
      "------------\n"
      "{context_msg}\n"
      "------------\n\n"
      "Using this as context, focus EXCLUSIVELY on refining your judgment of the ORIGINAL AITA conflict by following these rules:\n"
      "\t- If the SIMILAR conflict reveals stronger reasoning for a different classification: Change your judgment.\n"
      "\t- If the SIMILAR conflict provides additional support for your current classification: Enhance your reasoning.\n"
      "\t- If the SIMILAR conflict offers new perspectives: Incorporate them regardless of whether they change or support your classification.\n"
      "\t- If the SIMILAR conflict seems less relevant or compelling: Maintain your current judgment.\n"
      "You also MUST continue to follow these rules:\n"
      f"{AITA_rules}\n"
      "Your refined judgment: "
    )      
  }