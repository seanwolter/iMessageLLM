"""
Prompt templates for message analysis using Ollama.
All prompts are centralized here for easy modification and maintenance.
"""

def add_conversation_context(prompt, conversation_history):
    """Add conversation history context to a prompt if available"""
    if not conversation_history:
        return prompt
    
    context_section = "\n\n--- PREVIOUS CONVERSATION CONTEXT ---\n"
    context_section += "This is a continuation of a previous conversation. Here's what we discussed before:\n\n"
    
    for i, entry in enumerate(conversation_history, 1):
        context_section += f"Previous Question {i}: {entry['question']}\n"
        # Truncate long responses for context
        response = entry['response']
        if len(response) > 800:
            response = response[:800] + "...[truncated for context]"
        context_section += f"Previous Response {i}: {response}\n\n"
    
    context_section += "--- END PREVIOUS CONTEXT ---\n\n"
    context_section += "Please consider this previous context when answering the new question below.\n\n"
    
    return context_section + prompt

def get_single_conversation_prompt(conv_id, messages_text, question, message_count, start_date, end_date, conversation_history=None):
    """Prompt for analyzing a single conversation"""
    prompt = f"""You are analyzing a single conversation between two people. This conversation contains {message_count} messages spanning from {start_date} to {end_date}.

CONVERSATION #{conv_id} DATA (Format: DATE|SENDER|MESSAGE):
{messages_text}

ANALYSIS REQUEST: {question}

INSTRUCTIONS:
- This is CONVERSATION #{conv_id} - analyze it as a complete, discrete conversation unit
- Consider the conversation flow, themes, and relationship dynamics within this specific exchange
- Note the conversation's beginning, development, and conclusion
- Identify key topics, emotional tones, and interaction patterns
- Consider the time span and response patterns within this conversation
- Provide specific examples from the messages when relevant
- If this conversation relates to broader relationship patterns, note those connections
- Be concise and direct - provide essential insights without unnecessary elaboration
- Do not quote the conversation text unless absolutely necessary for your analysis

Please provide a focused, concise analysis of this individual conversation."""
    
    return add_conversation_context(prompt, conversation_history)

def get_conversation_segment_prompt(conv_id, question, seg_num, total_segments, start_date, end_date):
    """Prompt for analyzing a segment of a large conversation"""
    return f"""
CONVERSATION #{conv_id} - SEGMENT {seg_num}/{total_segments}:
{question}

This is part of a larger conversation. Focus on this segment while noting how it might connect to the broader conversation flow.
Date range: {start_date} to {end_date}
"""

def get_large_conversation_synthesis_prompt(conv_id, question, segment_responses):
    """Prompt for synthesizing segments back into a complete conversation analysis"""
    prompt = f"""
Synthesize the following segment analyses into a concise analysis of CONVERSATION #{conv_id}:

ORIGINAL QUESTION: {question}

SEGMENT ANALYSES:
"""
    
    for seg_resp in segment_responses:
        prompt += f"\n--- SEGMENT {seg_resp['segment']} ({seg_resp['date_range']}) - {seg_resp['message_count']} messages ---\n{seg_resp['response']}\n"
    
    prompt += f"""

SYNTHESIS TASK:
- Combine key insights from all segments into a unified analysis of Conversation #{conv_id}
- Maintain conversation flow and continuity across segments
- Identify overall patterns, themes, and development within this single conversation
- Provide a focused, concise answer about this specific conversation
- Avoid unnecessary elaboration and focus on essential insights
- Do not quote conversation text unless absolutely necessary for your analysis
"""
    
    return prompt

def get_chunk_synthesis_prompt(question, chunk_responses):
    """Prompt for synthesizing multiple conversation analyses within a chunk"""
    prompt = f"""
Synthesize the following individual conversation analyses into a concise understanding of this time period:

ORIGINAL QUESTION: {question}

INDIVIDUAL CONVERSATION ANALYSES:
"""
    
    for conv_resp in chunk_responses:
        prompt += f"\n--- CONVERSATION #{conv_resp['conversation_id']} ({conv_resp['date_range']}) - {conv_resp['message_count']} messages ---\n{conv_resp['response']}\n"
    
    prompt += f"""

SYNTHESIS TASK:
- Identify common themes and patterns across these {len(chunk_responses)} conversations
- Note the progression and development between conversations in this time period
- Highlight significant changes or consistency in communication patterns
- Provide integrated insights that address: {question}
- Be concise and focus on essential findings without unnecessary detail
- Do not quote conversation text unless absolutely necessary for your analysis
"""
    
    return prompt

def get_final_synthesis_prompt(question, responses, total_messages, total_conversations):
    """Prompt for final synthesis across all chunks/conversations"""
    prompt = f"""
Provide a concise final analysis based on the following conversation analyses:

ORIGINAL QUESTION: {question}

TOTAL SCOPE: {total_messages:,} messages across {total_conversations} conversations

CONVERSATION/CHUNK ANALYSES:
"""
    
    for resp in responses:
        if resp['type'] == 'single_conversation':
            conv = resp['conversation']
            prompt += f"\n--- CONVERSATION #{conv['conversation_id']} ({conv['date_range']}) - {conv['message_count']} messages ---\n{conv['response']}\n"
        elif resp['type'] == 'multiple_conversations':
            prompt += f"\n--- CHUNK {resp['chunk_num']} - {resp['conversation_count']} conversations, {resp['total_messages']} messages ---\n"
            if resp.get('synthesis'):
                prompt += f"{resp['synthesis']}\n"
            else:
                for conv in resp['conversations']:
                    prompt += f"Conversation #{conv['conversation_id']}: {conv['response']}\n"
        elif resp['type'] == 'large_conversation':
            conv = resp['conversation']
            prompt += f"\n--- LARGE CONVERSATION #{conv['conversation_id']} ({conv['date_range']}) - {conv['message_count']} messages ---\n{conv['response']}\n"
    
    prompt += f"""

FINAL SYNTHESIS TASK:
- Provide a concise, overarching analysis that addresses: {question}
- Integrate key insights from all {total_conversations} conversations analyzed
- Identify major themes, patterns, and evolution across the entire {total_messages:,} message dataset
- Note significant changes or developments over time
- Provide clear, actionable insights and conclusions in a focused manner
- Be direct and avoid unnecessary elaboration
- Do not quote conversation text unless absolutely necessary for your analysis
"""
    
    return prompt

def get_basic_analysis_prompt(messages_text, question, conversation_history=None):
    """Basic prompt for single-request analysis of smaller message sets"""
    prompt = f"""You are analyzing a conversation between two people. Below are chronologically ordered messages from their chat history.

CONVERSATION DATA (Format: DATE|SENDER|MESSAGE):
{messages_text}

ANALYSIS REQUEST: {question}

INSTRUCTIONS:
- Analyze the conversation patterns, themes, and context
- Consider the chronological progression and relationship dynamics
- Provide specific examples from the messages when relevant
- Be thoughtful and nuanced in your analysis
- If asked about emotions or relationships, consider both explicit and implicit cues
- Reference specific dates or time periods when relevant to your analysis
- Be concise and direct - focus on essential insights without unnecessary elaboration
- Do not quote the conversation text unless absolutely necessary for your analysis

Please provide a focused response based on the conversation data above."""
    
    return add_conversation_context(prompt, conversation_history)
