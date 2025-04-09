# Prompt Templates Used by the RAG Agent to Construct LLM Queries

The initial judgment and refinement prompt templates work together sequentially to generate the final response from the RAG agent. For example, when the agent retrieves five similar AITA conflicts to use as context, the process works as follows:
1. The initial judgment prompt template is used with the first retrieved conflict to create the initial response.
2. The refinement prompt template is then applied four times, once with each remaining conflict, to iteratively improve the response.

AITA Background Information and AITA Prompt Rules have been seperated out as constants that provide important background information to the agent regarding what the AITA subreddit is and how it functions.

## <span style="color: blue;">AITA Background Information</span>

AITA (Am I The A**hole) is a format where people share personal conflicts and ask for judgment.
In these situations:
- The writer describes a specific conflict or dilemma they're involved in.
- They explain their actions and the actions of other involved parties.
- They share relevant context like relationships, history, and circumstances.
- The core question is always about who's at fault in the conflict.
- Judgments focus on actions and choices, not on judging people as individuals.

## <span style="color: red;">AITA Prompt Rules</span>

You MUST follow these rules
- Your AITA classification choices are limited to you're the a\*\*hole (YTA) when the writer is causing the conflict, not the a\*\*hole (NTA) when someone other than the writer is causing the conflict, no a\*\*holes here (NAH) when no one is causing the conflict, and everyone sucks here (ESH) when everyone is causing the conflict.
- You MUST maintain consistency with the reasoning in the provided context and are NOT allowed to make independent moral judgments.
- You are NOT allowed to cite the specific details from the context examples to support your judgement.
- Use the context ONLY as a template for writing in a similar style (phrasing and expressions) and tone (casual vs formal), identifying the right amount of detail in justifications, and presenting your verdict strictly in the form of the AITA classification followed by the justification.
- Do NOT use racist, derogatory, or explicit language even if its used in the judgement example.

## Initial Judgement Prompt Template

__<span style="color: blue;">[AITA BACKGROUND INFORMATION]</span>__</br>
Study this NEW AITA conflict and judgement carefully:</br>
__{context_str}__</br>
Using the above as context, provide your initial judgment of this ORIGINAL AITA conflict:</br>
__{query_str}__</br>
You MUST follow these rules:</br>
- You MUST judge the ORIGINAL AITA conflict and NOT the NEW conflict.</br>
- __<span style="color: red;">[AITA PROMPT RULES]</span>__</br>

Your initial judgment:

## Refinement Prompt Template

__<span style="color: blue;">[AITA BACKGROUND INFORMATION]</span>__</br>
You are refining your judgment on this ORIGINAL AITA conflict:</br>
__{query_str}__</br>
This is your previous judgment of the ORIGINAL AITA conflict:</br>
__{existing_answer}__</br>
Now study this potentially SIMILAR AITA conflict and judgement:</br>
__{context_msg}__</br>
Using this as context, focus EXCLUSIVELY on refining your judgment of the ORIGINAL AITA conflict by following these rules:
- If the SIMILAR conflict reveals stronger reasoning for a different classification: Change your judgment.
- If the SIMILAR conflict provides additional support for your current classification: Enhance your reasoning.
- If the SIMILAR conflict offers new perspectives: Incorporate them regardless of whether they change or support your classification.
- If the SIMILAR conflict seems less relevant or compelling: Maintain your current judgment.</br>

You also MUST continue to follow these rules:</br>
- __<span style="color: red;">[AITA PROMPT RULES]</span>__</br>

Your refined judgment:
