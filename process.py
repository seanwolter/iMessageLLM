import csv
import re
import os
import time
import math
from collections import defaultdict
from datetime import datetime

from formatting_utils import (
    print_header, print_section, print_info, print_progress,
    print_success, print_error, print_warning, format_duration, format_file_size
)

def calculate_optimal_sample_size(total_size, confidence_level=0.95, margin_error=0.05):
    """Calculate statistically significant sample size for given population"""
    # Z-score for 95% confidence level
    z_score = 1.96 if confidence_level == 0.95 else 2.576
    
    # Conservative estimate assuming maximum variance (p=0.5)
    p = 0.5
    
    # Sample size calculation for finite population
    numerator = (z_score ** 2) * p * (1 - p)
    denominator = margin_error ** 2
    
    # Initial sample size (infinite population)
    initial_sample = numerator / denominator
    
    # Finite population correction
    if total_size <= initial_sample * 20:  # Apply correction if population is small
        sample_size = initial_sample / (1 + (initial_sample - 1) / total_size)
    else:
        sample_size = initial_sample
    
    return min(int(math.ceil(sample_size)), total_size)

def get_adaptive_sample_size(total_size, analysis_type="general"):
    """Get adaptive sample size based on dataset size and analysis type"""
    if total_size <= 100:
        return total_size  # Use all data for small datasets
    
    # Calculate statistically significant sample size
    stat_sample = calculate_optimal_sample_size(total_size)
    
    # Minimum sample sizes by analysis type
    min_samples = {
        "general": 1000,
        "conversation_quality": 2000,
        "momentum": 500,
        "context": 300
    }
    
    min_required = min_samples.get(analysis_type, 1000)
    
    # Use the larger of statistical requirement or minimum threshold
    recommended_size = max(stat_sample, min_required)
    
    # Cap at reasonable percentage of total data (max 10% for very large datasets)
    max_sample = min(total_size, max(recommended_size, int(total_size * 0.10)))
    
    return min(recommended_size, max_sample)

def get_adaptive_window_size(context_type="momentum", dataset_size=0):
    """Get adaptive window size based on context type and dataset characteristics"""
    base_windows = {
        "momentum": 5,
        "context": 3,
        "similarity": 4
    }
    
    base_size = base_windows.get(context_type, 5)
    
    # Scale window size based on dataset size for better analysis
    if dataset_size > 50000:
        return min(base_size + 3, 12)  # Larger windows for big datasets
    elif dataset_size > 10000:
        return min(base_size + 2, 10)
    elif dataset_size > 1000:
        return min(base_size + 1, 8)
    else:
        return base_size

def parse_timestamp(timestamp_text):
    """Parse timestamp text and return formatted date string"""
    timestamp_match = re.match(r'([^\(]+)', timestamp_text.strip())
    if timestamp_match:
        timestamp_str = timestamp_match.group(1).strip()
        try:
            dt = datetime.strptime(timestamp_str, '%b %d, %Y  %I:%M:%S %p')
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            return timestamp_str
    return timestamp_text.strip()

def is_conversation_starter(message):
    """Detect if a message is likely a conversation starter"""
    text = message.lower()
    starters = [
        'hey', 'hi', 'hello', 'good morning', 'good afternoon', 'good evening',
        'how are you', 'how\'s it going', 'what\'s up', 'what are you up to',
        'what\'re you up to', 'how was', 'how\'s your day', 'morning', 'evening'
    ]
    return any(starter in text for starter in starters)

def is_conversation_ender(message):
    """Detect if a message is likely a conversation ender"""
    text = message.lower()
    enders = [
        'goodnight', 'good night', 'gnight', 'night', 'bye', 'goodbye', 'talk later',
        'talk soon', 'see you', 'catch you later', 'ttyl', 'going to sleep', 'going to bed',
        'have fun', 'enjoy', 'sleep well', 'sweet dreams'
    ]
    return any(ender in text for ender in enders)

def get_dynamic_threshold(current_time, previous_time):
    """Calculate dynamic time threshold based on time of day and context"""
    current_hour = current_time.hour
    previous_hour = previous_time.hour
    
    # Night time (10 PM - 6 AM): shorter threshold
    if (current_hour >= 22 or current_hour <= 6) and (previous_hour >= 22 or previous_hour <= 6):
        return 3
    
    # Morning (6 AM - 10 AM): overnight gap threshold
    elif 6 <= current_hour <= 10:
        return 8
    
    # Day time (10 AM - 6 PM): active conversation time
    elif 10 <= current_hour <= 18:
        return 4
    
    # Evening (6 PM - 10 PM)
    else:
        return 5

def calculate_message_similarity(msg1, msg2):
    """Calculate semantic similarity between two messages using simple TF-IDF approach"""
    def get_words(text):
        words = re.findall(r'\w+', text.lower())
        return [word for word in words if len(word) > 2]
    
    words1 = set(get_words(msg1))
    words2 = set(get_words(msg2))
    
    if not words1 or not words2:
        return 0.0
    
    # Jaccard similarity
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0

def analyze_conversation_momentum(messages, index, window_size=None):
    """Analyze conversation momentum and engagement patterns"""
    if window_size is None:
        window_size = get_adaptive_window_size("momentum", len(messages))
    
    if index < window_size:
        return {'momentum': 'starting', 'engagement': 0.5, 'response_pattern': 'initial'}
    
    start_idx = max(0, index - window_size)
    window = messages[start_idx:index + 1]
    
    response_times = []
    for i in range(1, len(window)):
        current_time = datetime.strptime(window[i]['date'], '%Y-%m-%d %H:%M:%S')
        prev_time = datetime.strptime(window[i-1]['date'], '%Y-%m-%d %H:%M:%S')
        response_times.append((current_time - prev_time).total_seconds() / 60)
    
    avg_response_time = sum(response_times) / len(response_times) if response_times else 60
    turn_changes = sum(1 for i in range(1, len(window)) if window[i]['sender'] != window[i-1]['sender'])
    
    message_lengths = [len(msg['message']) for msg in window]
    length_variance = sum((l - sum(message_lengths)/len(message_lengths))**2 for l in message_lengths) / len(message_lengths)
    
    if avg_response_time < 5:
        momentum = 'high'
    elif avg_response_time < 30:
        momentum = 'medium'
    else:
        momentum = 'low'
    
    engagement = min(1.0, (turn_changes / window_size) * (1 / (1 + avg_response_time/60)) * (1 + length_variance/1000))
    
    return {
        'momentum': momentum,
        'engagement': engagement,
        'avg_response_time': avg_response_time,
        'turn_changes': turn_changes
    }

def detect_emotional_tone_shift(current_msg, previous_msg):
    """Detect significant emotional tone shifts that might indicate conversation boundaries"""
    positive_indicators = ['😀', '😃', '😄', '😁', '😊', '😍', '🥰', '😘', '💕', '❤️', '💜', 
                          'haha', 'lol', 'awesome', 'amazing', 'love', 'great', 'perfect', 'best']
    negative_indicators = ['😢', '😭', '😞', '😔', '😟', '😕', '😤', '😠', '😡', 
                          'sad', 'sorry', 'upset', 'angry', 'frustrated', 'terrible', 'worst', 'hate']
    
    def get_emotional_score(text):
        text_lower = text.lower()
        positive_count = sum(1 for indicator in positive_indicators if indicator in text_lower)
        negative_count = sum(1 for indicator in negative_indicators if indicator in text_lower)
        return positive_count - negative_count
    
    current_score = get_emotional_score(current_msg)
    previous_score = get_emotional_score(previous_msg)
    
    return abs(current_score - previous_score) >= 2

def analyze_context_window(messages, index, window_size=None):
    """Analyze context window for conversation continuity"""
    if window_size is None:
        window_size = get_adaptive_window_size("context", len(messages))
    
    start_idx = max(0, index - window_size)
    end_idx = min(len(messages), index + window_size + 1)
    window = messages[start_idx:end_idx]
    
    current_msg = messages[index]['message']
    topic_similarities = []
    
    for msg in window:
        if msg != messages[index]:
            similarity = calculate_message_similarity(current_msg, msg['message'])
            topic_similarities.append(similarity)
    
    avg_similarity = sum(topic_similarities) / len(topic_similarities) if topic_similarities else 0
    
    response_gaps = []
    for i in range(1, len(window)):
        current_time = datetime.strptime(window[i]['date'], '%Y-%m-%d %H:%M:%S')
        prev_time = datetime.strptime(window[i-1]['date'], '%Y-%m-%d %H:%M:%S')
        gap_minutes = (current_time - prev_time).total_seconds() / 60
        response_gaps.append(gap_minutes)
    
    if len(response_gaps) >= 2:
        recent_gaps = response_gaps[-2:]
        avg_recent_gap = sum(recent_gaps) / len(recent_gaps)
        pattern_break = response_gaps[-1] > avg_recent_gap * 3
    else:
        pattern_break = False
    
    return {
        'topic_similarity': avg_similarity,
        'pattern_break': pattern_break,
        'avg_response_gap': sum(response_gaps) / len(response_gaps) if response_gaps else 0
    }

def detect_activity_transitions(current_msg, previous_msg):
    """Detect activity or location transitions that indicate conversation boundaries"""
    activity_indicators = [
        'at', 'in', 'going to', 'arrived at', 'made it to', 'just got to',
        'now', 'just', 'about to', 'going to', 'starting', 'finished',
        'work', 'home', 'school', 'class', 'meeting', 'dinner', 'lunch', 'breakfast',
        'shopping', 'driving', 'walking', 'running', 'gym', 'party', 'bar', 'restaurant'
    ]
    
    current_lower = current_msg.lower()
    previous_lower = previous_msg.lower()
    
    current_activities = [ind for ind in activity_indicators if ind in current_lower]
    previous_activities = [ind for ind in activity_indicators if ind in previous_lower]
    
    return len(current_activities) > 0 and len(set(current_activities) - set(previous_activities)) > 0

def detect_topic_change(current_msg, previous_msg):
    """Enhanced topic change detection using multiple signals"""
    if not current_msg or not previous_msg:
        return False
    
    current_text = current_msg.lower()
    previous_text = previous_msg.lower()
    
    if '?' in current_text and '?' not in previous_text:
        question_words = ['what', 'when', 'where', 'why', 'how', 'who', 'which']
        if any(word in current_text for word in question_words):
            return True
    
    similarity = calculate_message_similarity(current_msg, previous_msg)
    if similarity < 0.1:
        return True
    
    if detect_emotional_tone_shift(current_msg, previous_msg):
        return True
    
    if detect_activity_transitions(current_msg, previous_msg):
        return True
    
    return False

def assign_conversation_ids(messages, base_gap_hours=6):
    """Assign conversation IDs based on message timing and content patterns"""
    if not messages:
        return messages
    
    print_header("CONVERSATION ANALYSIS")
    print("Analyzing conversation patterns...")
    
    # Sort messages by date to ensure chronological order
    print(f"Sorting {len(messages):,} messages chronologically...")
    messages.sort(key=lambda msg: datetime.strptime(msg['date'], '%Y-%m-%d %H:%M:%S'))
    
    conversation_id = 1
    messages[0]['conversation_id'] = conversation_id
    conversation_stats = {
        'time_based': 0, 'starter_based': 0, 'ender_based': 0, 'sender_change': 0,
        'momentum_based': 0, 'topic_change': 0, 'emotional_shift': 0, 'activity_transition': 0,
        'pattern_break': 0, 'context_analysis': 0
    }
    
    start_time = time.time()
    print("Grouping conversations...")
    
    for i in range(1, len(messages)):
        current_time = datetime.strptime(messages[i]['date'], '%Y-%m-%d %H:%M:%S')
        previous_time = datetime.strptime(messages[i-1]['date'], '%Y-%m-%d %H:%M:%S')
        current_msg = messages[i]['message']
        previous_msg = messages[i-1]['message']
        current_sender = messages[i]['sender']
        previous_sender = messages[i-1]['sender']
        
        time_gap = current_time - previous_time
        time_gap_hours = time_gap.total_seconds() / 3600
        time_gap_minutes = time_gap.total_seconds() / 60
        
        # Get dynamic threshold based on time of day
        dynamic_threshold = get_dynamic_threshold(current_time, previous_time)
        
        # Analyze conversation patterns
        momentum_analysis = analyze_conversation_momentum(messages, i)
        context_analysis = analyze_context_window(messages, i)
        topic_changed = detect_topic_change(current_msg, previous_msg)
        
        should_start_new_conversation = False
        reason = ""
        confidence = 0.0  # Confidence score for the decision
        
        # === PRIMARY CONVERSATION BREAKS ===
        
        # 1. Major time gap (always break)
        if time_gap_hours > dynamic_threshold:
            should_start_new_conversation = True
            reason = "time_based"
            confidence = 1.0
            conversation_stats['time_based'] += 1
        
        # 2. Strong conversation starter after reasonable gap
        elif time_gap_hours > 2 and is_conversation_starter(current_msg):
            should_start_new_conversation = True
            reason = "starter_based"
            confidence = 0.9
            conversation_stats['starter_based'] += 1
        
        # 3. Previous message was clear conversation ender
        elif time_gap_hours > 1.5 and is_conversation_ender(previous_msg):
            should_start_new_conversation = True
            reason = "ender_based"
            confidence = 0.9
            conversation_stats['ender_based'] += 1
        
        # === SECONDARY CONVERSATION BREAKS ===
        
        # 4. Low momentum conversation with time gap
        elif (time_gap_hours > 2.5 and momentum_analysis['momentum'] == 'low' and 
              momentum_analysis['engagement'] < 0.3):
            should_start_new_conversation = True
            reason = "momentum_based"
            confidence = 0.8
            conversation_stats['momentum_based'] += 1
        
        # 5. Significant topic change with moderate gap
        elif time_gap_hours > 1 and topic_changed:
            should_start_new_conversation = True
            reason = "topic_change"
            confidence = 0.7
            conversation_stats['topic_change'] += 1
        
        # 6. Context pattern break (conversation rhythm disruption)
        elif time_gap_hours > 1.5 and context_analysis['pattern_break']:
            should_start_new_conversation = True
            reason = "pattern_break"
            confidence = 0.7
            conversation_stats['pattern_break'] += 1
        
        # 7. Emotional tone shift with gap
        elif (time_gap_hours > 1 and 
              detect_emotional_tone_shift(current_msg, previous_msg)):
            should_start_new_conversation = True
            reason = "emotional_shift"
            confidence = 0.6
            conversation_stats['emotional_shift'] += 1
        
        # 8. Activity/location transition
        elif (time_gap_hours > 1 and 
              detect_activity_transitions(current_msg, previous_msg)):
            should_start_new_conversation = True
            reason = "activity_transition"
            confidence = 0.6
            conversation_stats['activity_transition'] += 1
        
        # === CONTEXTUAL ANALYSIS ===
        
        # 9. Low topic similarity in context window
        elif (time_gap_hours > 2 and context_analysis['topic_similarity'] < 0.15):
            should_start_new_conversation = True
            reason = "context_analysis"
            confidence = 0.6
            conversation_stats['context_analysis'] += 1
        
        # 10. Sender change with extended gap and low engagement
        elif (time_gap_hours > 3 and current_sender != previous_sender and 
              momentum_analysis['engagement'] < 0.4):
            should_start_new_conversation = True
            reason = "sender_change"
            confidence = 0.5
            conversation_stats['sender_change'] += 1
        
        # === RAPID EXCHANGE PRESERVATION ===
        # Don't break conversations during rapid exchanges (< 5 minutes)
        # unless there's a very strong indicator (confidence > 0.9)
        if time_gap_minutes < 5 and confidence < 0.9:
            should_start_new_conversation = False
            reason = "preserved_rapid_exchange"
        
        # === HIGH ENGAGEMENT PRESERVATION ===
        # Don't break high-engagement conversations too easily
        if momentum_analysis['engagement'] > 0.8 and confidence < 0.8:
            should_start_new_conversation = False
            reason = "preserved_high_engagement"
        
        if should_start_new_conversation:
            conversation_id += 1
        
        messages[i]['conversation_id'] = conversation_id
        
        # Store analysis metadata for debugging/improvement
        messages[i]['_analysis'] = {
            'momentum': momentum_analysis,
            'context': context_analysis,
            'decision_reason': reason,
            'decision_confidence': confidence,
            'time_gap_hours': time_gap_hours
        }
        
        # Show progress every 10000 messages
        if (i + 1) % 10000 == 0:
            progress_pct = (i + 1) / len(messages) * 100
            print(f"Analyzed {i + 1:,} messages ({progress_pct:.0f}%)")
    
    print()  # New line after progress
    
    elapsed_time = time.time() - start_time
    print_success(f"Assigned {conversation_id:,} conversations in {format_duration(elapsed_time)}")
    
    print_section("Decision Statistics")
    total_decisions = sum(conversation_stats.values())
    for reason, count in sorted(conversation_stats.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            percentage = (count / total_decisions * 100) if total_decisions > 0 else 0
            reason_formatted = reason.replace('_', ' ').title()
            print_info(f"{reason_formatted}", f"{count} ({percentage:.1f}%)")
    
    # Clean up analysis metadata before returning (required for CSV compatibility)
    for msg in messages:
        if '_analysis' in msg:
            del msg['_analysis']
    
    return messages

def extract_messages(html_file):
    """Extract messages from HTML file using regex for speed"""
    print_header("MESSAGE EXTRACTION")
    
    file_size = os.path.getsize(html_file)
    print_info("Input file", html_file)
    print_info("File size", format_file_size(file_size))
    
    message_pattern = re.compile(
        r'<div class="message">.*?<span class="timestamp">(.*?)</span>.*?<span class="sender">(.*?)</span>.*?<span class="bubble">(.*?)</span>.*?</div>',
        re.DOTALL
    )
    
    messages = []
    processed_bytes = 0
    start_time = time.time()
    
    print("\nStarting HTML parsing and message extraction...")
    
    with open(html_file, 'r', encoding='utf-8') as file:
        chunk_size = 1024 * 1024
        buffer = ""
        
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
                
            buffer += chunk
            processed_bytes += len(chunk)
            
            matches = list(message_pattern.finditer(buffer))
            
            for match in matches:
                try:
                    timestamp = match.group(1).strip()
                    sender = match.group(2).strip()
                    message_text = match.group(3).strip()
                    
                    if timestamp and sender and message_text:
                         formatted_date = parse_timestamp(timestamp)
                         
                         if sender == "Me":
                             standardized_sender = "Me"
                         else:
                             standardized_sender = "Them"
                         
                         messages.append({
                             'date': formatted_date,
                             'sender': standardized_sender,
                             'message': message_text
                         })
                        
                except Exception as e:
                    print_error(f"parsing message: {e}")
                    continue
            
            if matches:
                last_match_end = matches[-1].end()
                buffer = buffer[last_match_end:]
            
            progress_pct = processed_bytes / file_size * 100
            print(f"Processed {progress_pct:.0f}% - {len(messages):,} messages found")
    
    print()  # New line after progress
    elapsed_time = time.time() - start_time
    print_success(f"Extracted {len(messages):,} messages in {format_duration(elapsed_time)}")
    return messages

def save_to_csv(messages, output_file):
    """Save messages to CSV file with progress tracking"""
    print_section("CSV Export")
    print_info("Output file", output_file)
    print_info("Messages to write", len(messages))
    
    start_time = time.time()
    print("\nWriting messages to CSV...")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['date', 'sender', 'message', 'conversation_id']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for i, message in enumerate(messages):
            writer.writerow(message)
            
            if (i + 1) % 5000 == 0 or (i + 1) == len(messages):
                progress_pct = (i + 1) / len(messages) * 100
                print(f"Written {i + 1:,} messages ({progress_pct:.0f}%)")
    
    print()  # New line after progress
    elapsed_time = time.time() - start_time
    file_size = os.path.getsize(output_file)
    print_success(f"CSV saved: {format_file_size(file_size)} in {format_duration(elapsed_time)}")



def analyze_conversations(messages):
    """Analyze conversation statistics with enhanced metrics"""
    if not messages:
        return
    
    print_header("CONVERSATION ANALYSIS")
    
    conversations = {}
    for msg in messages:
        conv_id = msg.get('conversation_id', 1)
        if conv_id not in conversations:
            conversations[conv_id] = []
        conversations[conv_id].append(msg)
    
    print_info("Total conversations", len(conversations))
    print_info("Total messages", len(messages))
    
    lengths = []
    durations = []
    turn_exchanges = []
    
    for conv in conversations.values():
        lengths.append(len(conv))
        
        if len(conv) > 1:
            start_time = datetime.strptime(conv[0]['date'], '%Y-%m-%d %H:%M:%S')
            end_time = datetime.strptime(conv[-1]['date'], '%Y-%m-%d %H:%M:%S')
            duration_hours = (end_time - start_time).total_seconds() / 3600
            durations.append(duration_hours)
            
            turns = 1
            for i in range(1, len(conv)):
                if conv[i]['sender'] != conv[i-1]['sender']:
                    turns += 1
            turn_exchanges.append(turns)
    
    print_section("Message Statistics")
    avg_msgs = sum(lengths) / len(lengths) if lengths else 0
    print_info("Average messages per conversation", f"{avg_msgs:.1f}")
    print_info("Shortest conversation", f"{min(lengths) if lengths else 0} messages")
    print_info("Longest conversation", f"{max(lengths) if lengths else 0} messages")
    print_info("Single message conversations", sum(1 for l in lengths if l == 1))
    print_info("Short conversations (2-10 msgs)", sum(1 for l in lengths if 2 <= l <= 10))
    print_info("Long conversations (11+ msgs)", sum(1 for l in lengths if l > 10))
    
    if durations:
        print_section("Duration Statistics")
        avg_duration = sum(durations) / len(durations)
        print_info("Average conversation duration", f"{avg_duration:.1f} hours")
        print_info("Shortest conversation", f"{min(durations):.1f} hours")
        print_info("Longest conversation", f"{max(durations):.1f} hours")
        print_info("Quick conversations (<1 hour)", sum(1 for d in durations if d < 1))
        print_info("Extended conversations (1-6 hours)", sum(1 for d in durations if 1 <= d <= 6))
        print_info("Marathon conversations (6+ hours)", sum(1 for d in durations if d > 6))
    
    if turn_exchanges:
        print_section("Interaction Statistics")
        avg_turns = sum(turn_exchanges) / len(turn_exchanges)
        print_info("Average turn exchanges", f"{avg_turns:.1f}")
        print_info("Monologue conversations (1 turn)", sum(1 for t in turn_exchanges if t == 1))
        print_info("Interactive conversations (2+ turns)", sum(1 for t in turn_exchanges if t >= 2))
    
    print_section("Notable Conversations")
    sorted_convs = sorted(conversations.items(), key=lambda x: len(x[1]), reverse=True)
    
    print("Longest Conversations:")
    for conv_id, conv in sorted_convs[:3]:
        start_date = conv[0]['date']
        end_date = conv[-1]['date']
        duration = (datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S') - 
                   datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')).total_seconds() / 3600
        print(f"  #{conv_id}: {len(conv):,} messages, {duration:.1f}h ({start_date[:10]})")
    
    print("\nRecent Conversations:")
    recent_convs = sorted(conversations.items(), key=lambda x: x[1][-1]['date'], reverse=True)
    for conv_id, conv in recent_convs[:5]:
        start_date = conv[0]['date']
        end_date = conv[-1]['date']
        print(f"  #{conv_id}: {len(conv):,} messages ({start_date[:10]} → {end_date[:10]})")

def load_messages(csv_file):
    """Load messages from CSV file and ensure chronological order"""
    print_header("LOADING MESSAGES")
    print_info("Input file", csv_file)
    
    try:
        start_time = time.time()
        with open(csv_file, 'r', encoding='utf-8') as file:
            messages = list(csv.DictReader(file))
        
        messages.sort(key=lambda msg: msg['date'])
        
        elapsed_time = time.time() - start_time
        file_size = os.path.getsize(csv_file)
        
        print_success(f"Loaded {len(messages):,} messages from {format_file_size(file_size)} file in {format_duration(elapsed_time)}")
        
        if messages and 'conversation_id' in messages[0]:
            analyze_conversations(messages)
        
        return messages
    except Exception as e:
        print_error(f"loading CSV: {e}")
        return []


def analyze_conversation_quality(messages, sample_size=None):
    """Analyze the quality of conversation grouping"""
    import copy
    
    if sample_size is None:
        sample_size = get_adaptive_sample_size(len(messages), "conversation_quality")
    
    if len(messages) < sample_size:
        sample_size = len(messages)
    
    print_header("CONVERSATION QUALITY ANALYSIS")
    print_info("Dataset size", f"{len(messages):,} messages")
    print_info("Sample size", f"{sample_size:,} messages ({sample_size/len(messages)*100:.1f}%)")
    
    # Validate sample size adequacy
    min_required = calculate_optimal_sample_size(len(messages))
    if sample_size < min_required:
        print_warning(f"Sample size may be too small for statistical significance (recommended: {min_required:,})")
    else:
        print_info("Statistical significance", "✓ Adequate sample size")
    
    sample_messages = copy.deepcopy(messages[:sample_size])
    
    # Run conversation grouping with analysis
    print("Running conversation grouping on sample...")
    
    if not sample_messages:
        return
    
    sample_messages.sort(key=lambda msg: datetime.strptime(msg['date'], '%Y-%m-%d %H:%M:%S'))
    
    conversation_id = 1
    sample_messages[0]['conversation_id'] = conversation_id
    ai_conversations = defaultdict(list)
    ai_conversations[conversation_id].append(sample_messages[0])
    break_reasons = defaultdict(int)
    confidence_scores = []
    
    for i in range(1, len(sample_messages)):
        current_time = datetime.strptime(sample_messages[i]['date'], '%Y-%m-%d %H:%M:%S')
        previous_time = datetime.strptime(sample_messages[i-1]['date'], '%Y-%m-%d %H:%M:%S')
        current_msg = sample_messages[i]['message']
        previous_msg = sample_messages[i-1]['message']
        
        time_gap_hours = (current_time - previous_time).total_seconds() / 3600
        dynamic_threshold = get_dynamic_threshold(current_time, previous_time)
        
        should_start_new_conversation = False
        reason = ""
        confidence = 0.0
        
        if time_gap_hours > dynamic_threshold:
            should_start_new_conversation = True
            reason = "time_based"
            confidence = 1.0
        elif time_gap_hours > 2 and is_conversation_starter(current_msg):
            should_start_new_conversation = True
            reason = "starter_based"
            confidence = 0.9
        elif time_gap_hours > 1.5 and is_conversation_ender(previous_msg):
            should_start_new_conversation = True
            reason = "ender_based"
            confidence = 0.9
        elif time_gap_hours > 1 and detect_topic_change(current_msg, previous_msg):
            should_start_new_conversation = True
            reason = "topic_change"
            confidence = 0.7
        
        if should_start_new_conversation:
            conversation_id += 1
            if reason:
                break_reasons[reason] += 1
                confidence_scores.append(confidence)
        
        sample_messages[i]['conversation_id'] = conversation_id
        ai_conversations[conversation_id].append(sample_messages[i])
    
    ai_messages = sample_messages
    
    print_section("Decision Analysis")
    print_info("Conversations created", len(ai_conversations))
    if confidence_scores:
        avg_confidence = sum(confidence_scores)/len(confidence_scores)
        print_info("Average decision confidence", f"{avg_confidence:.2f}")
    else:
        print_warning("No confidence data available")
    
    print_section("Break Reason Distribution")
    total_breaks = sum(break_reasons.values())
    for reason, count in sorted(break_reasons.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_breaks * 100) if total_breaks > 0 else 0
        reason_formatted = reason.replace('_', ' ').title()
        print_info(reason_formatted, f"{count} ({percentage:.1f}%)")
    
    rapid_exchanges = 0
    high_engagement_convs = 0
    topic_coherent_convs = 0
    
    for conv_msgs in ai_conversations.values():
        if len(conv_msgs) < 2:
            continue
            
        has_rapid = False
        total_engagement = 0
        topic_similarities = []
        
        for i in range(1, len(conv_msgs)):
            current_time = datetime.strptime(conv_msgs[i]['date'], '%Y-%m-%d %H:%M:%S')
            prev_time = datetime.strptime(conv_msgs[i-1]['date'], '%Y-%m-%d %H:%M:%S')
            gap_minutes = (current_time - prev_time).total_seconds() / 60
            
            if gap_minutes < 5:
                has_rapid = True
            
            similarity = calculate_message_similarity(conv_msgs[i]['message'], conv_msgs[i-1]['message'])
            topic_similarities.append(similarity)
        
        if has_rapid:
            rapid_exchanges += 1
        
        turn_changes = sum(1 for i in range(1, len(conv_msgs)) if conv_msgs[i]['sender'] != conv_msgs[i-1]['sender'])
        engagement_score = turn_changes / len(conv_msgs) if len(conv_msgs) > 1 else 0
        
        if engagement_score > 0.5:
            high_engagement_convs += 1
        
        avg_similarity = sum(topic_similarities) / len(topic_similarities) if topic_similarities else 0
        if avg_similarity > 0.2:
            topic_coherent_convs += 1
    
    print_section("Conversation Quality Metrics")
    total_convs = len(ai_conversations)
    if total_convs > 0:
        print_info("Rapid exchange conversations", f"{rapid_exchanges} ({rapid_exchanges/total_convs*100:.1f}%)")
        print_info("High engagement conversations", f"{high_engagement_convs} ({high_engagement_convs/total_convs*100:.1f}%)")
        print_info("Topic coherent conversations", f"{topic_coherent_convs} ({topic_coherent_convs/total_convs*100:.1f}%)")
    
    print_section("Decision Summary")
    if break_reasons:
        print("  Most common decision types:")
        for reason, count in sorted(break_reasons.items(), key=lambda x: x[1], reverse=True)[:3]:
            reason_title = reason.replace('_', ' ').title()
            percentage = (count / sum(break_reasons.values()) * 100) if sum(break_reasons.values()) > 0 else 0
            print(f"    {reason_title}: {count} decisions ({percentage:.1f}%)")
    else:
        print("  No conversation breaks detected in sample.")

def analyze_existing_data(csv_file='data/messages.csv'):
    """Analyze existing CSV data without reprocessing"""
    print_header("ANALYZING EXISTING DATA")
    
    messages = load_messages(csv_file)
    
    if messages and 'conversation_id' in messages[0]:
        analyze_conversation_quality(messages)
        
        print_header("ANALYSIS COMPLETE")
        print("\nAnalysis complete!")
    else:
        print_error("No conversation IDs found in data. Please run full processing first.")

def main():
    """Convert HTML messages to CSV with conversation IDs"""
    print_header("iMESSAGE PROCESSOR")
    print("Starting message processing pipeline...")
    
    start_time = time.time()
    
    messages = extract_messages('data/message.html')
    messages = assign_conversation_ids(messages)
    save_to_csv(messages, 'data/messages.csv')
    
    total_time = time.time() - start_time
    
    print_header("PROCESSING COMPLETE")
    print_success(f"Pipeline completed in {format_duration(total_time)}")
    print_info("Final output", "data/messages.csv")
    print_info("Total messages processed", len(messages))
    
    if messages and 'conversation_id' in messages[0]:
        total_conversations = max(int(msg['conversation_id']) for msg in messages)
        print_info("Total conversations", total_conversations)
        
        analyze_conversations(messages)
        analyze_conversation_quality(messages)
    
    print_header("ANALYSIS COMPLETE")
    print("\nAll done! Your messages are ready for analysis.")

if __name__ == "__main__":
    main()
