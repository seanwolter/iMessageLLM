import argparse
import json
import requests
import tiktoken
import os
import hashlib
from datetime import datetime
from deepseek_tokenizer import deepseek_tokenizer
from process import load_messages
from prompts import (
    get_single_conversation_prompt,
    get_conversation_segment_prompt,
    get_large_conversation_synthesis_prompt,
    get_chunk_synthesis_prompt,
    get_final_synthesis_prompt,
    get_basic_analysis_prompt
)

MODEL = "deepseek-r1:latest"
MAX_CONTEXT_TOKENS = 128000

def compress_message_format(messages):
    """Use more compact message format to save tokens"""
    return "\n".join([
        f"{msg['date']}|{msg['sender']}|{msg['message']}"
        for msg in messages
    ])

def count_tokens(text, use_deepseek=True):
    """Count the number of tokens in a text string using appropriate tokenizer"""
    if use_deepseek:
        try:
            # Use pre-initialized DeepSeek tokenizer for accurate token counting
            tokens = deepseek_tokenizer.encode(text)
            return len(tokens)
        except Exception as e:
            print(f"DeepSeek tokenizer failed, using tiktoken: {e}")
            use_deepseek = False
    
    if not use_deepseek:
        try:
            # Fallback to tiktoken cl100k_base (will be less accurate for DeepSeek)
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            return len(tokens)
        except Exception as e:
            # Final fallback to word count approximation
            print(f"Token counting failed, using word approximation: {e}")
            words = len(text.split())
            return int(words * 1.3)  # Rough approximation: 1 token ≈ 0.75 words

def print_token_stats(messages, question):
    """Print token usage statistics"""
    total_message_text = compress_message_format(messages)
    total_message_tokens = count_tokens(total_message_text)
    question_tokens = count_tokens(question)
    
    print(f"\nToken Analysis:")
    print(f"  Messages: {len(messages):,} ({total_message_tokens:,} tokens)")
    print(f"  Question: {question_tokens:,} tokens")
    print(f"  Average per message: {total_message_tokens / len(messages):.1f}")
    print(f"  Context usage: {(total_message_tokens / MAX_CONTEXT_TOKENS) * 100:.1f}% of {MAX_CONTEXT_TOKENS:,}")

def get_conversation_file_path(conversation_id=None):
    """Generate conversation history file path"""
    conversations_dir = "conversations"
    os.makedirs(conversations_dir, exist_ok=True)
    
    if conversation_id:
        filename = f"conversation_{conversation_id}.json"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.json"
    
    return os.path.join(conversations_dir, filename)

def save_conversation_history(question, response, conversation_data, file_path=None):
    """Save conversation history to file"""
    conversation_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "response": response,
        "conversation_data": conversation_data,
        "model": MODEL
    }
    
    # Load existing conversation or create new one
    conversation_history = []
    
    # If no file path specified, auto-generate one
    if not file_path:
        file_path = get_conversation_file_path()
    
    # If file exists, load it to continue the conversation
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                conversation_history = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            conversation_history = []
    
    conversation_history.append(conversation_entry)
    
    # Save updated conversation
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(conversation_history, f, indent=2, ensure_ascii=False)
    
    print(f"Conversation saved to: {file_path}")
    return file_path

def load_conversation_history(file_path):
    """Load conversation history from file"""
    if not os.path.exists(file_path):
        print(f"Conversation file not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error reading conversation file: {e}")
        return None

def list_conversation_files():
    """List available conversation history files"""
    conversations_dir = "conversations"
    if not os.path.exists(conversations_dir):
        print("No conversation history found")
        return []
    
    files = []
    for filename in os.listdir(conversations_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(conversations_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                
                if history:
                    first_entry = history[0]
                    last_entry = history[-1]
                    files.append({
                        'filename': filename,
                        'path': file_path,
                        'entries': len(history),
                        'first_timestamp': first_entry.get('timestamp', 'Unknown'),
                        'last_timestamp': last_entry.get('timestamp', 'Unknown'),
                        'first_question': first_entry.get('question', 'Unknown')[:80] + ('...' if len(first_entry.get('question', '')) > 80 else '')
                    })
            except (json.JSONDecodeError, FileNotFoundError):
                continue
    
    # Sort by last timestamp
    files.sort(key=lambda x: x['last_timestamp'], reverse=True)
    return files

def format_conversation_context(history):
    """Format conversation history for context in new prompts"""
    if not history:
        return ""
    
    context_parts = []
    for i, entry in enumerate(history, 1):
        context_parts.append(f"Previous Question {i}: {entry['question']}")
        
        # Truncate long responses for context
        response = entry['response']
        if len(response) > 500:
            response = response[:500] + "...[truncated]"
        context_parts.append(f"Previous Response {i}: {response}")
    
    return "\n\n".join(context_parts)

def get_cache_directory():
    """Get or create cache directory"""
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def generate_cache_key(conv_id, question, messages_hash, model):
    """Generate a unique cache key for a conversation analysis"""
    # Create a hash of the key components
    key_data = f"{conv_id}_{question}_{messages_hash}_{model}"
    cache_key = hashlib.md5(key_data.encode('utf-8')).hexdigest()
    return cache_key

def get_messages_hash(messages):
    """Generate a hash of the messages content to detect changes"""
    messages_text = compress_message_format(messages)
    return hashlib.md5(messages_text.encode('utf-8')).hexdigest()

def get_cache_file_path(cache_key):
    """Get the file path for a cache entry"""
    cache_dir = get_cache_directory()
    return os.path.join(cache_dir, f"{cache_key}.json")

def save_to_cache(cache_key, question, response, conversation_data, messages_hash):
    """Save analysis result to cache"""
    cache_data = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "response": response,
        "conversation_data": conversation_data,
        "messages_hash": messages_hash,
        "model": MODEL
    }
    
    cache_file = get_cache_file_path(cache_key)
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)
    
    return cache_file

def load_from_cache(cache_key, current_messages_hash):
    """Load analysis result from cache if valid"""
    cache_file = get_cache_file_path(cache_key)
    
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        # Verify the messages haven't changed
        if cache_data.get('messages_hash') != current_messages_hash:
            print(f"Cache invalid: messages changed since last analysis")
            return None
        
        # Verify model matches
        if cache_data.get('model') != MODEL:
            print(f"Cache invalid: model changed from {cache_data.get('model')} to {MODEL}")
            return None
        
        return cache_data
        
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Cache file corrupted: {e}")
        return None

def clear_cache():
    """Clear all cached results"""
    cache_dir = get_cache_directory()
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
    
    for cache_file in cache_files:
        os.remove(os.path.join(cache_dir, cache_file))
    
    print(f"Cleared {len(cache_files)} cached results")

def list_cache_entries():
    """List all cached analysis results"""
    cache_dir = get_cache_directory()
    if not os.path.exists(cache_dir):
        print("No cache directory found")
        return []
    
    cache_files = []
    for filename in os.listdir(cache_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(cache_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                cache_files.append({
                    'filename': filename,
                    'path': file_path,
                    'timestamp': cache_data.get('timestamp', 'Unknown'),
                    'question': cache_data.get('question', 'Unknown')[:80] + ('...' if len(cache_data.get('question', '')) > 80 else ''),
                    'conversation_data': cache_data.get('conversation_data', {}),
                    'model': cache_data.get('model', 'Unknown')
                })
            except (json.JSONDecodeError, FileNotFoundError):
                continue
    
    # Sort by timestamp
    cache_files.sort(key=lambda x: x['timestamp'], reverse=True)
    return cache_files

def group_messages_by_conversation(messages):
    """Group messages by conversation_id for conversation-based analysis"""
    if not messages:
        return {}
    
    conversations = {}
    for message in messages:
        conv_id = message.get('conversation_id', '1')
        if conv_id not in conversations:
            conversations[conv_id] = []
        conversations[conv_id].append(message)
    
    # Sort each conversation chronologically
    for conv_id in conversations:
        conversations[conv_id].sort(key=lambda msg: msg['date'])
    
    return conversations

def filter_conversations_by_criteria(conversations, min_messages=1, max_messages=None, 
                                   include_ids=None, exclude_ids=None):
    """Filter conversations based on various criteria"""
    filtered = {}
    
    for conv_id, messages in conversations.items():
        # Convert conv_id to string for comparison
        conv_id_str = str(conv_id)
        
        # Skip if not in include list (if specified)
        if include_ids and conv_id_str not in [str(id) for id in include_ids]:
            continue
            
        # Skip if in exclude list
        if exclude_ids and conv_id_str in [str(id) for id in exclude_ids]:
            continue
            
        # Check message count criteria
        msg_count = len(messages)
        if msg_count < min_messages:
            continue
        if max_messages and msg_count > max_messages:
            continue
            
        filtered[conv_id] = messages
    
    return filtered

def create_conversation_chunks(conversations, question, max_context_tokens=MAX_CONTEXT_TOKENS):
    """Create chunks based on conversations, respecting token limits"""
    if not conversations:
        return []
    
    # Reserve tokens for prompt overhead
    prompt_overhead = 1000
    question_tokens = count_tokens(question)
    synthesis_buffer = 2000
    
    max_chunk_tokens = max_context_tokens - prompt_overhead - question_tokens - synthesis_buffer
    
    chunks = []
    current_chunk_conversations = []
    current_tokens = 0
    
    # Sort conversations by their start date
    sorted_conversations = sorted(conversations.items(), 
                                key=lambda x: x[1][0]['date'])
    
    print(f"Chunking {len(conversations)} conversations (max {max_chunk_tokens:,} tokens/chunk)")
    
    for conv_id, conv_messages in sorted_conversations:
        # Calculate tokens for this conversation
        conv_text = compress_message_format(conv_messages)
        conv_tokens = count_tokens(conv_text)
        
        # If this single conversation exceeds the limit, handle it specially
        if conv_tokens > max_chunk_tokens:
            # If we have conversations in current chunk, save it first
            if current_chunk_conversations:
                chunks.append({
                    'type': 'multiple_conversations',
                    'conversations': current_chunk_conversations.copy(),
                    'token_count': current_tokens
                })
                current_chunk_conversations = []
                current_tokens = 0
            
            # Create a special chunk for this oversized conversation
            chunks.append({
                'type': 'large_conversation',
                'conversations': [{'id': conv_id, 'messages': conv_messages}],
                'token_count': conv_tokens,
                'needs_splitting': True
            })
            continue
        
        # If adding this conversation would exceed the limit, save current chunk
        if current_tokens + conv_tokens > max_chunk_tokens and current_chunk_conversations:
            chunks.append({
                'type': 'multiple_conversations',
                'conversations': current_chunk_conversations.copy(),
                'token_count': current_tokens
            })
            current_chunk_conversations = []
            current_tokens = 0
        
        # Add conversation to current chunk
        current_chunk_conversations.append({
            'id': conv_id,
            'messages': conv_messages
        })
        current_tokens += conv_tokens
    
    # Add the last chunk if it has conversations
    if current_chunk_conversations:
        chunks.append({
            'type': 'multiple_conversations',
            'conversations': current_chunk_conversations,
            'token_count': current_tokens
        })
    
    return chunks

def create_conversation_aware_chunks(messages, question, max_context_tokens=MAX_CONTEXT_TOKENS):
    """Create chunks based on conversations, respecting token limits"""
    if not messages:
        return []
    
    if 'conversation_id' not in messages[0]:
        raise ValueError("Messages must have conversation_id field. Run 'python process.py' first to process your data.")
    
    print("Using conversation chunking")
    conversations = group_messages_by_conversation(messages)
    return create_conversation_chunks(conversations, question, max_context_tokens)

def ask_ollama_single_conversation(conversation_messages, conv_id, question, model=MODEL, save_history=True, history_file=None, conversation_history=None, force_reprocess=False):
    """Analyze a single conversation"""
    if not conversation_messages:
        return None
    
    # Get conversation metadata
    start_date = conversation_messages[0]['date']
    end_date = conversation_messages[-1]['date']
    message_count = len(conversation_messages)
    
    # Check cache first (unless force reprocessing)
    messages_hash = get_messages_hash(conversation_messages)
    cache_key = generate_cache_key(conv_id, question, messages_hash, model)
    
    if not force_reprocess:
        cached_result = load_from_cache(cache_key, messages_hash)
        if cached_result:
            print(f"Conversation #{conv_id}: Using cached result (~{cached_result.get('conversation_data', {}).get('token_count', 'Unknown')} tokens, {start_date} to {end_date}) [CACHED]")
            
            # Still save to conversation history if requested
            if save_history:
                conversation_data = cached_result.get('conversation_data', {})
                save_conversation_history(question, cached_result['response'], conversation_data, history_file)
            
            return cached_result['response']
    
    # Format conversation for analysis
    messages_text = compress_message_format(conversation_messages)
    
    # Create conversation-specific prompt
    prompt = get_single_conversation_prompt(conv_id, messages_text, question, message_count, start_date, end_date, conversation_history)

    token_count = count_tokens(prompt)
    print(f"Conversation #{conv_id}: {message_count} messages (~{token_count:,} tokens, {start_date} to {end_date})")
    
    try:
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": True, "keep_alive": -1},
            timeout=None, stream=True
        )
        response.raise_for_status()
        
        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line.decode('utf-8'))
                    if 'response' in chunk:
                        text = chunk['response']
                        print(text, end='', flush=True)
                        full_response += text
                        if chunk.get('done', False):
                            print(f"\n[Conv #{conv_id} complete]")
                            break
                except json.JSONDecodeError:
                    continue
        
        # Save to cache and conversation history if we got a response
        if full_response:
            conversation_data = {
                'type': 'single_conversation',
                'conversation_id': conv_id,
                'message_count': message_count,
                'date_range': f"{start_date} to {end_date}",
                'token_count': token_count
            }
            
            # Save to cache
            save_to_cache(cache_key, question, full_response, conversation_data, messages_hash)
            
            # Save to conversation history if requested
            if save_history:
                save_conversation_history(question, full_response, conversation_data, history_file)
        
        return full_response
        
    except Exception as e:
        print(f"Error in conversation #{conv_id}: {e}")
        return None

def ask_ollama_chunked(messages, question, model=MODEL, save_history=True, history_file=None, conversation_history=None, force_reprocess=False):
    """Ask Ollama about messages using conversation-based chunking to preserve conversation boundaries"""
    if not messages:
        print("No messages provided")
        return None
    
    # Ensure messages are in chronological order
    messages = sorted(messages, key=lambda msg: msg['date'])
    
    # Create conversation-aware chunks
    chunks = create_conversation_aware_chunks(messages, question)
    
    if not chunks:
        print("No chunks created")
        return None
    
    # Conversation-based processing
    responses = []
    total_conversations = sum(len(chunk['conversations']) for chunk in chunks)
    total_messages = len(messages)
    
    print(f"Processing {total_messages:,} messages in {total_conversations} conversations ({len(chunks)} chunks)")
    
    for chunk_num, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {chunk_num}/{len(chunks)} ---")
        
        if chunk['type'] == 'multiple_conversations':
            # Process multiple conversations in this chunk
            print(f"Chunk: {len(chunk['conversations'])} conversations (~{chunk['token_count']:,} tokens)")
            
            chunk_responses = []
            for conv_data in chunk['conversations']:
                conv_id = conv_data['id']
                conv_messages = conv_data['messages']
                
                print(f"\nAnalyzing conversation #{conv_id}...")
                conv_response = ask_ollama_single_conversation(conv_messages, conv_id, question, model, save_history=False, history_file=None, conversation_history=conversation_history, force_reprocess=force_reprocess)
                
                if conv_response:
                    chunk_responses.append({
                        'conversation_id': conv_id,
                        'message_count': len(conv_messages),
                        'date_range': f"{conv_messages[0]['date']} to {conv_messages[-1]['date']}",
                        'response': conv_response
                    })
            
            # Synthesize this chunk if multiple conversations
            if len(chunk_responses) > 1:
                print(f"\nSynthesizing chunk {chunk_num} ({len(chunk_responses)} conversations)...")
                chunk_synthesis = synthesize_chunk_conversations(chunk_responses, question, model)
                responses.append({
                    'chunk_num': chunk_num,
                    'type': 'multiple_conversations',
                    'conversations': chunk_responses,
                    'synthesis': chunk_synthesis,
                    'conversation_count': len(chunk_responses),
                    'total_messages': sum(r['message_count'] for r in chunk_responses)
                })
            elif len(chunk_responses) == 1:
                # Single conversation in chunk
                responses.append({
                    'chunk_num': chunk_num,
                    'type': 'single_conversation',
                    'conversation': chunk_responses[0],
                    'conversation_count': 1,
                    'total_messages': chunk_responses[0]['message_count']
                })
        
        elif chunk['type'] == 'large_conversation':
            # Handle oversized single conversation
            conv_data = chunk['conversations'][0]
            conv_id = conv_data['id']
            conv_messages = conv_data['messages']
            
            print(f"Large conversation #{conv_id} (~{chunk['token_count']:,} tokens) - splitting")
            
            if chunk.get('needs_splitting', False):
                # Split large conversation into segments
                conv_response = ask_ollama_large_conversation(conv_messages, conv_id, question, model)
            else:
                conv_response = ask_ollama_single_conversation(conv_messages, conv_id, question, model, save_history=False, history_file=None, conversation_history=conversation_history, force_reprocess=force_reprocess)
            
            if conv_response:
                responses.append({
                    'chunk_num': chunk_num,
                    'type': 'large_conversation',
                    'conversation': {
                        'conversation_id': conv_id,
                        'message_count': len(conv_messages),
                        'date_range': f"{conv_messages[0]['date']} to {conv_messages[-1]['date']}",
                        'response': conv_response
                    },
                    'conversation_count': 1,
                    'total_messages': len(conv_messages),
                    'was_split': chunk.get('needs_splitting', False)
                })
    
    # Final synthesis across all chunks
    if len(responses) > 1:
        print(f"\n--- Final Synthesis ---")
        
        final_synthesis = synthesize_all_responses(responses, question, model, total_messages, total_conversations)
        result = {
            'individual_responses': responses,
            'final_synthesis': final_synthesis,
            'summary': {
                'total_messages': total_messages,
                'total_conversations': total_conversations,
                'chunks_processed': len(responses)
            }
        }
        
        # Save to conversation history if requested
        if save_history and final_synthesis:
            conversation_data = {
                'type': 'chunked_analysis',
                'total_messages': total_messages,
                'total_conversations': total_conversations,
                'chunks_processed': len(responses)
            }
            save_conversation_history(question, final_synthesis, conversation_data, history_file)
        
        return result
    elif len(responses) == 1:
        # Single response, no need for synthesis
        print(f"\nAnalysis complete: {total_messages:,} messages, {total_conversations} conversations")
        result = responses[0]
        
        # Save to conversation history if requested
        if save_history:
            # Extract response text for saving
            response_text = ""
            if result.get('type') == 'single_conversation' and result.get('conversation'):
                response_text = result['conversation'].get('response', '')
            elif result.get('synthesis'):
                response_text = result['synthesis']
            
            if response_text:
                conversation_data = {
                    'type': 'single_chunk_analysis',
                    'total_messages': total_messages,
                    'total_conversations': total_conversations,
                    'chunks_processed': 1
                }
                save_conversation_history(question, response_text, conversation_data, history_file)
        
        return result
    else:
        print("No responses generated")
        return None


def ask_ollama_large_conversation(conv_messages, conv_id, question, model):
    """Handle conversations that exceed token limits by splitting them into segments"""
    print(f"Splitting conversation #{conv_id} into segments...")
    
    # Split conversation into smaller segments while preserving chronological order
    temp_chunks = []
    prompt_overhead = 1000
    question_tokens = count_tokens(question)
    synthesis_buffer = 2000
    max_chunk_tokens = MAX_CONTEXT_TOKENS - prompt_overhead - question_tokens - synthesis_buffer
    
    current_chunk = []
    current_tokens = 0
    
    for message in conv_messages:
        message_text = f"{message['date']}|{message['sender']}|{message['message']}"
        message_tokens = count_tokens(message_text)
        
        if current_tokens + message_tokens > max_chunk_tokens and current_chunk:
            temp_chunks.append(current_chunk.copy())
            current_chunk = []
            current_tokens = 0
        
        current_chunk.append(message)
        current_tokens += message_tokens
    
    if current_chunk:
        temp_chunks.append(current_chunk)
    
    # Analyze each segment
    segment_responses = []
    for seg_num, segment in enumerate(temp_chunks, 1):
        print(f"\nSegment {seg_num}/{len(temp_chunks)} of conversation #{conv_id}...")
        
        segment_question = get_conversation_segment_prompt(
            conv_id, question, seg_num, len(temp_chunks), 
            segment[0]['date'], segment[-1]['date']
        )
        
        segment_response = ask_ollama_single(segment, segment_question, model)
        if segment_response:
            segment_responses.append({
                'segment': seg_num,
                'date_range': f"{segment[0]['date']} to {segment[-1]['date']}",
                'message_count': len(segment),
                'response': segment_response
            })
    
    # Synthesize segments back into conversation analysis
    if len(segment_responses) > 1:
        print(f"\nSynthesizing conversation #{conv_id} from {len(segment_responses)} segments...")
        
        synthesis_prompt = get_large_conversation_synthesis_prompt(conv_id, question, segment_responses)
        
        synthesis_response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": synthesis_prompt, "stream": True, "keep_alive": -1},
            timeout=None, stream=True
        )
        
        synthesized_response = ""
        if synthesis_response.status_code == 200:
            for line in synthesis_response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        if 'response' in chunk:
                            text = chunk['response']
                            print(text, end='', flush=True)
                            synthesized_response += text
                        if chunk.get('done', False):
                            print(f"\n[Conversation #{conv_id} synthesis complete]")
                            break
                    except json.JSONDecodeError:
                        continue
        
        return synthesized_response
    elif len(segment_responses) == 1:
        return segment_responses[0]['response']
    else:
        return None

def synthesize_chunk_conversations(chunk_responses, question, model):
    """Synthesize multiple conversation analyses within a chunk"""
    if len(chunk_responses) <= 1:
        return None
    
    synthesis_prompt = get_chunk_synthesis_prompt(question, chunk_responses)
    
    synthesis_response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": synthesis_prompt, "stream": True, "keep_alive": -1},
        timeout=None, stream=True
    )
    
    synthesized_text = ""
    if synthesis_response.status_code == 200:
        for line in synthesis_response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line.decode('utf-8'))
                    if 'response' in chunk:
                        text = chunk['response']
                        print(text, end='', flush=True)
                        synthesized_text += text
                    if chunk.get('done', False):
                        print(f"\n[Chunk synthesis complete]")
                        break
                except json.JSONDecodeError:
                    continue
    
    return synthesized_text

def synthesize_all_responses(responses, question, model, total_messages, total_conversations):
    """Final synthesis across all chunks/conversations"""
    synthesis_prompt = get_final_synthesis_prompt(question, responses, total_messages, total_conversations)
    
    print(f"Generating final synthesis from {len(responses)} units...")
    
    synthesis_response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": synthesis_prompt, "stream": True, "keep_alive": -1},
        timeout=None, stream=True
    )
    
    final_synthesis = ""
    if synthesis_response.status_code == 200:
        for line in synthesis_response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line.decode('utf-8'))
                    if 'response' in chunk:
                        text = chunk['response']
                        print(text, end='', flush=True)
                        final_synthesis += text
                        if chunk.get('done', False):
                            print(f"\n\nFinal synthesis complete: {total_messages:,} messages, {total_conversations} conversations")
                            break
                except json.JSONDecodeError:
                    continue
    
    return final_synthesis

def ask_ollama_single(messages, question, model=MODEL, conversation_history=None):
    """Original single-request approach for smaller message sets"""
    # Format messages for prompt - use compact format to save tokens
    messages_text = compress_message_format(messages)
    
    prompt = get_basic_analysis_prompt(messages_text, question, conversation_history)

    # Count tokens in the prompt using DeepSeek tokenizer
    token_count = count_tokens(prompt)
    
    print(f"Analyzing {len(messages)} messages ({token_count:,} tokens) with {model}")
    print(f"Question: {question}")
    print("Response:")
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": True, "keep_alive": -1},
            timeout=None, stream=True
        )
        response.raise_for_status()
        
        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line.decode('utf-8'))
                    if 'response' in chunk:
                        text = chunk['response']
                        print(text, end='', flush=True)
                        full_response += text
                    if chunk.get('done', False):
                        duration = chunk.get('total_duration', 'N/A')
                        print(f"\n\nComplete ({duration}s)")
                        break
                except json.JSONDecodeError:
                    continue
        
        return full_response
        
    except requests.exceptions.ConnectionError:
        print("Ollama not running on localhost:11434")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def ask_ollama(messages, question, model=MODEL, save_history=True, history_file=None, conversation_history=None, force_reprocess=False):
    """Ask Ollama a question about messages using conversation-based analysis"""
    if not messages:
        print("No messages provided")
        return None
    
    if 'conversation_id' not in messages[0]:
        raise ValueError("Messages must have conversation_id field. Run 'python process.py' first to process your data.")
    
    # Show detailed token analysis
    print_token_stats(messages, question)
    
    # Calculate tokens for the entire message set
    messages_text = compress_message_format(messages)
    question_tokens = count_tokens(question)
    prompt_overhead = 1000  # Conservative estimate for prompt formatting
    
    total_tokens_needed = count_tokens(messages_text) + question_tokens + prompt_overhead
    
    print(f"Tokens needed: {total_tokens_needed:,} / {MAX_CONTEXT_TOKENS:,} (90% threshold: {int(MAX_CONTEXT_TOKENS * 0.9):,})")
    
    # Group messages by conversation first
    conversations = group_messages_by_conversation(messages)
    
    # If small enough for single analysis and only one conversation, use single approach
    if (total_tokens_needed <= MAX_CONTEXT_TOKENS * 0.9 and len(conversations) == 1):
        print("Using direct analysis (single conversation)")
        conv_id = list(conversations.keys())[0]
        conv_messages = conversations[conv_id]
        return ask_ollama_single_conversation(conv_messages, conv_id, question, model, save_history, history_file, conversation_history, force_reprocess)
    else:
        print("Using conversation chunking (multiple conversations or large dataset)")
        return ask_ollama_chunked(messages, question, model, save_history, history_file, conversation_history, force_reprocess)

def filter_by_years(messages, years):
    """Filter messages by years and return selected messages in chronological order"""
    if not years:
        return messages
    
    selected = []
    for year in years:
        year_messages = [msg for msg in messages if msg['date'].startswith(str(year))]
        selected.extend(year_messages)
        print(f"Year {year}: {len(year_messages):,} messages")
    
    # Sort selected messages chronologically to maintain proper order
    # when multiple years are selected
    if selected:
        selected.sort(key=lambda msg: msg['date'])
        return selected
    else:
        return messages


def analyze_individual_conversation(conv_id, csv_file='data/messages.csv', question=None, model=MODEL, save_history=True, history_file=None, conversation_history=None, force_reprocess=False):
    """Analyze a single conversation by ID"""
    print(f"Loading conversation #{conv_id}...")
    messages = load_messages(csv_file)
    if not messages:
        print("No messages found. Run 'python process.py' first.")
        return
    
    # Group by conversation
    conversations = group_messages_by_conversation(messages)
    
    if str(conv_id) not in conversations:
        print(f"Conversation #{conv_id} not found")
        available_ids = sorted([int(cid) for cid in conversations.keys() if cid.isdigit()])[:10]
        print(f"Available IDs (first 10): {available_ids}")
        return
    
    conv_messages = conversations[str(conv_id)]
    print(f"Conversation #{conv_id}: {len(conv_messages)} messages ({conv_messages[0]['date']} to {conv_messages[-1]['date']})")
    
    if not question:
        question = "Analyze this conversation. What are the main topics, themes, and relationship dynamics?"
    
    # Analyze the single conversation
    return ask_ollama_single_conversation(conv_messages, conv_id, question, model, save_history, history_file, conversation_history, force_reprocess)

def list_conversations(csv_file='data/messages.csv', limit=20):
    """List available conversations with metadata"""
    print("Loading conversations...")
    messages = load_messages(csv_file)
    if not messages:
        print("No messages found.")
        return
    
    conversations = group_messages_by_conversation(messages)
    
    # Calculate conversation stats
    conv_stats = []
    for conv_id, conv_messages in conversations.items():
        conv_stats.append({
            'id': conv_id,
            'message_count': len(conv_messages),
            'start_date': conv_messages[0]['date'],
            'end_date': conv_messages[-1]['date'],
            'duration_hours': (
                datetime.strptime(conv_messages[-1]['date'], '%Y-%m-%d %H:%M:%S') - 
                datetime.strptime(conv_messages[0]['date'], '%Y-%m-%d %H:%M:%S')
            ).total_seconds() / 3600 if len(conv_messages) > 1 else 0
        })
    
    # Sort by various criteria
    print(f"\nFound {len(conversations)} conversations (showing top {limit}):\n")
    
    # Longest conversations
    print("Longest conversations:")
    longest = sorted(conv_stats, key=lambda x: x['message_count'], reverse=True)[:limit]
    for conv in longest:
        print(f"  #{conv['id']}: {conv['message_count']:,} messages ({conv['start_date'][:10]} to {conv['end_date'][:10]})")
    
    # Most recent conversations
    print(f"\nMost recent:")
    recent = sorted(conv_stats, key=lambda x: x['end_date'], reverse=True)[:limit]
    for conv in recent:
        print(f"  #{conv['id']}: {conv['message_count']:,} messages (ended {conv['end_date'][:10]})")
    
    # Longest duration conversations
    print(f"\nLongest duration:")
    duration = sorted(conv_stats, key=lambda x: x['duration_hours'], reverse=True)[:limit]
    for conv in duration:
        hours = conv['duration_hours']
        if hours > 24:
            duration_str = f"{hours/24:.1f} days"
        else:
            duration_str = f"{hours:.1f} hours"
        print(f"  #{conv['id']}: {conv['message_count']:,} messages over {duration_str}")

def main():
    """Ask questions about messages using Ollama"""
    parser = argparse.ArgumentParser(
        description="Ask questions about messages using Ollama with conversation-based analysis",
        epilog="""Examples:
  # Analyze all messages from specific years
  python ask_messages.py --year 2017 --question "What are the main themes?"
  python ask_messages.py --years 2017 2018 --question "How did our relationship change?"
  
  # Analyze specific conversations
  python ask_messages.py --conversation 42 --question "What happened in this conversation?"
  python ask_messages.py --conversations 10 15 20 --question "Compare these conversations"
  
  # Filter conversations by characteristics
  python ask_messages.py --min-messages 50 --question "What are themes in longer conversations?"
  python ask_messages.py --max-messages 10 --question "What are quick exchanges about?"
  
  # Conversation history management (automatically saves by default)
  python ask_messages.py --list-history  # List saved conversations
  python ask_messages.py --question "What are the main themes?"  # Auto-saves with timestamp
  python ask_messages.py --question "What are themes?" --save-conversation my_analysis.json  # Custom filename
  python ask_messages.py --load-conversation my_analysis.json --question "Tell me more about those themes"
  python ask_messages.py --question "Quick analysis" --no-save  # Don't save this one
  
  # Cache management (speeds up repeated analyses)
  python ask_messages.py --list-cache  # List cached results
  python ask_messages.py --question "What are themes?" --force-reprocess  # Ignore cache, reprocess
  python ask_messages.py --clear-cache  # Clear all cached results
  
  # List available conversations
  python ask_messages.py --list-conversations
  
  # General analysis
  python ask_messages.py --question "What are common topics?" --csv custom.csv
"""
    )
    
    # Existing arguments
    parser.add_argument('--year', type=int, help='Single year to filter (e.g., 2017)')
    parser.add_argument('--years', type=int, nargs='+', help='Multiple years (e.g., 2017 2018)')
    parser.add_argument('--question', help='Question to ask about messages (required unless --list-conversations)')
    parser.add_argument('--csv', default='data/messages.csv', help='CSV file to load (default: data/messages.csv)')
    parser.add_argument('--model', default=MODEL, help='Ollama model to use')
    
    # New conversation-based arguments
    parser.add_argument('--conversation', type=int, help='Analyze single conversation by ID')
    parser.add_argument('--conversations', type=int, nargs='+', help='Analyze specific conversations by IDs')
    parser.add_argument('--min-messages', type=int, help='Filter conversations with at least N messages')
    parser.add_argument('--max-messages', type=int, help='Filter conversations with at most N messages')
    parser.add_argument('--list-conversations', action='store_true', help='List available conversations with metadata')
    
    # Conversation history arguments
    parser.add_argument('--save-conversation', help='Save conversation to specific file (default: auto-generated)')
    parser.add_argument('--load-conversation', help='Load and continue previous conversation from file')
    parser.add_argument('--list-history', action='store_true', help='List available conversation history files')
    parser.add_argument('--no-save', action='store_true', help='Disable automatic conversation saving')
    
    # Cache management arguments
    parser.add_argument('--force-reprocess', action='store_true', help='Force reprocessing even if cached results exist')
    parser.add_argument('--list-cache', action='store_true', help='List cached analysis results')
    parser.add_argument('--clear-cache', action='store_true', help='Clear all cached results')
    
    args = parser.parse_args()
    
    # Handle cache management
    if args.clear_cache:
        clear_cache()
        return
    
    if args.list_cache:
        cache_entries = list_cache_entries()
        if cache_entries:
            print(f"Found {len(cache_entries)} cached analysis results:\n")
            for i, entry in enumerate(cache_entries, 1):
                conv_data = entry['conversation_data']
                print(f"{i}. {entry['filename']}")
                print(f"   Question: {entry['question']}")
                print(f"   Timestamp: {entry['timestamp'][:19]}")
                print(f"   Model: {entry['model']}")
                if conv_data.get('conversation_id'):
                    print(f"   Conversation: #{conv_data['conversation_id']} ({conv_data.get('message_count', 'Unknown')} messages)")
                print(f"   Path: {entry['path']}\n")
        return
    
    # Handle listing conversation history
    if args.list_history:
        files = list_conversation_files()
        if files:
            print(f"Found {len(files)} conversation history files:\n")
            for i, file_info in enumerate(files, 1):
                print(f"{i}. {file_info['filename']}")
                print(f"   Entries: {file_info['entries']}")
                print(f"   Last: {file_info['last_timestamp'][:19]}")
                print(f"   First question: {file_info['first_question']}")
                print(f"   Path: {file_info['path']}\n")
        return
    
    # Handle listing conversations
    if args.list_conversations:
        list_conversations(args.csv)
        return
    
    # Handle loading previous conversation
    previous_history = None
    if args.load_conversation:
        previous_history = load_conversation_history(args.load_conversation)
        if not previous_history:
            return
        print(f"Loaded conversation history with {len(previous_history)} entries")
        
        # Show previous conversation context
        print("\n--- Previous Conversation Context ---")
        for i, entry in enumerate(previous_history, 1):
            print(f"{i}. Q: {entry['question']}")
            response_preview = entry['response'][:200] + ('...' if len(entry['response']) > 200 else '')
            print(f"   A: {response_preview}\n")
    
    # Require question for analysis
    if not args.question:
        if args.load_conversation:
            print("Error: --question required to continue conversation")
        else:
            print("Error: --question required (use --list-conversations to browse)")
        return
    
    # Handle single conversation analysis
    if args.conversation:
        # Use loaded conversation file to continue, or custom save file, or auto-generate
        history_file = args.load_conversation or args.save_conversation
        analyze_individual_conversation(args.conversation, args.csv, args.question, args.model, 
                                      not args.no_save, history_file, previous_history, args.force_reprocess)
        return
    
    # Load messages
    print("Loading messages...")
    messages = load_messages(args.csv)
    if not messages:
        print("No messages found. Run 'python process.py' first.")
        return
    
    print(f"Loaded {len(messages):,} messages")
    
    # Ensure we have conversation IDs
    if 'conversation_id' not in messages[0]:
        print("Error: Messages need conversation IDs. Run 'python process.py' first.")
        return
    
    # Apply conversation-based filtering
    if args.conversations or args.min_messages or args.max_messages:
        conversations = group_messages_by_conversation(messages)
        
        # Filter conversations by criteria
        filtered_conversations = filter_conversations_by_criteria(
            conversations,
            min_messages=args.min_messages or 1,
            max_messages=args.max_messages,
            include_ids=args.conversations
        )
        
        # Convert back to message list
        selected_messages = []
        for conv_messages in filtered_conversations.values():
            selected_messages.extend(conv_messages)
        
        selected_messages.sort(key=lambda msg: msg['date'])
        
        print(f"Filtered: {len(selected_messages):,} messages in {len(filtered_conversations)} conversations")
    else:
        # Filter by years (original functionality)
        years = []
        if args.year:
            years.append(args.year)
        if args.years:
            years.extend(args.years)
        
        if years:
            years = sorted(set(years))  # Remove duplicates and sort
            selected_messages = filter_by_years(messages, years)
            print(f"Selected: {len(selected_messages):,} messages from {years}")
        else:
            selected_messages = messages
            print("Using all messages")
    
    if not selected_messages:
        print("No messages found for specified criteria")
        return
    
    # Determine conversation history settings
    save_history = not args.no_save
    # Use loaded conversation file to continue, or custom save file, or auto-generate
    history_file = args.load_conversation or args.save_conversation
    
    # Ask Ollama
    ask_ollama(selected_messages, args.question, args.model, save_history, history_file, previous_history, args.force_reprocess)

if __name__ == "__main__":
    main()
