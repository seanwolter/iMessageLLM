# iMessage LLM Processor

A powerful Python tool for converting iMessage HTML exports (created with [imessage-exporter](https://github.com/ReagentX/imessage-exporter)) into structured, analyzable CSV data with intelligent conversation grouping and AI-powered analysis using Large Language Models.

## Features

### 🔧 Core Processing
- **HTML to CSV Conversion**: Extract messages from iMessage HTML exports into structured CSV format
- **Intelligent Conversation Grouping**: Automatically groups messages into conversations using advanced algorithms
- **Progress Tracking**: Real-time progress bars and performance metrics during processing
- **Memory Efficient**: Processes large HTML files using streaming with chunk-based parsing

### 🧠 Smart Conversation Detection
The tool uses multiple sophisticated algorithms to intelligently group messages into conversations:

- **Dynamic Time Thresholds**: Adaptive time gaps based on time of day (night: 3h, morning: 8h, day: 4h, evening: 5h)
- **Content Analysis**: Detects conversation starters ("hey", "hello") and enders ("goodnight", "bye")
- **Momentum Analysis**: Analyzes response times, turn-taking patterns, and engagement levels
- **Topic Change Detection**: Uses semantic similarity to identify topic shifts
- **Emotional Tone Shifts**: Detects significant changes in emotional tone using emoji and keyword analysis
- **Activity Transitions**: Identifies location/activity changes that indicate conversation boundaries
- **Context Windows**: Analyzes surrounding messages for conversation continuity

### 📊 Advanced Analytics
- **Conversation Statistics**: Detailed metrics on conversation length, duration, and interaction patterns
- **Quality Analysis**: Evaluates conversation grouping quality with confidence scores
- **Sample-Based Analysis**: Uses statistically significant sampling for large datasets
- **Performance Metrics**: Tracks processing time and provides optimization insights

### 🎨 Rich Terminal Output
- **Colored Output**: Beautiful, readable terminal output with progress indicators
- **Structured Reporting**: Organized sections with clear headers and formatting
- **Error Handling**: Graceful error handling with informative messages

### 🤖 AI-Powered Analysis
- **LLM Integration**: Uses Ollama with DeepSeek-R1 for intelligent message analysis
- **Conversation Context**: Maintains conversation boundaries during analysis
- **Smart Caching**: Caches results to speed up repeated analyses
- **Session Management**: Save and continue analysis sessions

## Quick Start

### Step 0: Export Your iMessage Data

Before using this tool, you need to export your iMessage conversations to HTML format. We recommend using [imessage-exporter](https://github.com/ReagentX/imessage-exporter):

```bash
# Install imessage-exporter
brew install imessage-exporter

# Export all messages to HTML format
imessage-exporter --format html --output ./data/

# Or export specific conversations
imessage-exporter --format html --output ./data/ --filter-contact "John Doe"
```

The exporter will create HTML files in your specified output directory. Rename or move the desired export to `data/message.html` for processing.

**Note**: imessage-exporter requires macOS and access to your Messages database. For more export options and detailed instructions, see the [imessage-exporter documentation](https://github.com/ReagentX/imessage-exporter).

### Steps 1-5: Process and Analyze

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install and start Ollama
# Download from https://ollama.ai, then:
ollama serve
ollama pull deepseek-r1:14b

# 3. Prepare your data directory and move export
mkdir -p data
mv ~/path/to/export.html data/message.html

# 4. Process messages
python process.py

# 5. Ask questions about your messages
python ask_messages.py --question "What are the main themes in our conversations?"
```

## Prerequisites

- **macOS**: Required for accessing iMessage database
- **Python 3.7+**: For running the processing and analysis scripts
- **Homebrew**: For installing imessage-exporter (optional but recommended)
- **Ollama**: For running the AI analysis locally

## Installation

### 1. Export Your iMessage Data

Install and use [imessage-exporter](https://github.com/ReagentX/imessage-exporter) to export your messages:

```bash
# Install via Homebrew (macOS)
brew install imessage-exporter

# Or install via Cargo if you have Rust
cargo install imessage-exporter
```

### 2. Clone This Repository

```bash
git clone <repository-url>
cd imessage_llm
```

### 3. Create Virtual Environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Process Your Messages

First, ensure you have exported your iMessage data to HTML format. You can use [imessage-exporter](https://github.com/ReagentX/imessage-exporter) for this:

```bash
# Export messages to HTML (if not done already)
imessage-exporter --format html --output ./data/
```

Then run the processing pipeline to convert HTML messages to structured CSV:

```bash
python process.py
```

This will:
1. Extract messages from `data/message.html`
2. Apply intelligent conversation grouping
3. Save results to `data/messages.csv`
4. Display comprehensive analytics

**Input File Requirements**: 
- Place your iMessage HTML export in the `data/` directory as `message.html`
- The tool expects HTML format exported by imessage-exporter or similar tools
- The HTML should contain message containers with timestamps, senders, and message content

### Step 2: Ask Questions About Your Messages

After processing, use the AI-powered analysis tool to ask questions:

```bash
python ask_messages.py --question "What are the main themes in our conversations?"
```

This tool uses Ollama with DeepSeek-R1 to provide intelligent analysis of your message history.

## AI-Powered Message Analysis (ask_messages.py)

### Features
- **LLM-based Analysis**: Uses Ollama with DeepSeek-R1 model for intelligent conversation analysis
- **Conversation-aware**: Analyzes individual conversations or groups maintaining context
- **Smart Chunking**: Automatically handles large datasets by splitting them intelligently
- **Response Caching**: Caches analysis results for faster repeated queries
- **Conversation History**: Saves and continues analysis sessions [[memory:8547400]]
- **Token Optimization**: Uses compact message format and DeepSeek tokenizer for efficient processing

### Usage Examples

#### Basic Analysis
```bash
# Ask a general question about all messages
python ask_messages.py --question "What are the main themes in our conversations?"

# Analyze specific years
python ask_messages.py --year 2017 --question "What happened this year?"
python ask_messages.py --years 2017 2018 2019 --question "How did our relationship evolve?"
```

#### Conversation-Specific Analysis
```bash
# List available conversations
python ask_messages.py --list-conversations

# Analyze a single conversation
python ask_messages.py --conversation 42 --question "What is this conversation about?"

# Analyze multiple conversations
python ask_messages.py --conversations 10 15 20 --question "Compare these conversations"

# Filter by conversation length
python ask_messages.py --min-messages 50 --question "What are themes in longer conversations?"
python ask_messages.py --max-messages 10 --question "What are quick exchanges about?"
```

#### Conversation History Management
```bash
# List saved conversation histories
python ask_messages.py --list-history

# Continue a previous analysis session
python ask_messages.py --load-conversation my_analysis.json --question "Tell me more about that"

# Save to specific file (auto-saves by default)
python ask_messages.py --question "Analyze themes" --save-conversation analysis_2024.json

# Disable auto-saving for one-off queries
python ask_messages.py --question "Quick check" --no-save
```

#### Cache Management
```bash
# List cached results
python ask_messages.py --list-cache

# Force reprocessing (ignore cache)
python ask_messages.py --question "Analyze again" --force-reprocess

# Clear all cached results
python ask_messages.py --clear-cache
```

### Requirements
- **Ollama**: Must be running on `localhost:11434`
- **DeepSeek-R1 Model**: Install with `ollama pull deepseek-r1:14b`
- **Processed Data**: Run `process.py` first to generate `data/messages.csv`

## Message Processing (process.py)

### Features
- **HTML to CSV Conversion**: Extracts messages from iMessage HTML exports
- **Intelligent Conversation Grouping**: Uses multiple algorithms to detect conversation boundaries
- **Performance Optimized**: Handles multi-gigabyte files with streaming and chunked processing
- **Statistical Analysis**: Provides detailed metrics on conversations and grouping quality

### Advanced Usage

Functions can be imported and used programmatically:

```python
from process import extract_messages, assign_conversation_ids, save_to_csv, analyze_conversations

# Extract messages from HTML
messages = extract_messages('data/message.html')

# Group into conversations
messages_with_conversations = assign_conversation_ids(messages)

# Save to CSV
save_to_csv(messages_with_conversations, 'data/output.csv')

# Analyze results
analyze_conversations(messages_with_conversations)
```

### Key Functions

#### process.py Functions

##### `extract_messages(html_file)`
- Extracts messages from HTML file using regex parsing
- Returns list of message dictionaries with date, sender, and message content
- Handles large files efficiently with chunked processing

##### `assign_conversation_ids(messages, base_gap_hours=6)`
- Intelligently groups messages into conversations
- Uses multiple detection algorithms for optimal accuracy
- Returns messages with added `conversation_id` field

##### `analyze_conversations(messages)`
- Provides comprehensive conversation analytics
- Shows statistics on message counts, durations, turn exchanges
- Identifies notable conversations and patterns

##### `analyze_conversation_quality(messages, sample_size=None)`
- Evaluates the quality of conversation grouping
- Uses statistical sampling for large datasets
- Provides confidence scores and decision analysis

##### `analyze_existing_data(csv_file='data/messages.csv')`
- Analyzes pre-processed CSV data without reprocessing
- Useful for exploring already converted data

#### ask_messages.py Functions

##### `ask_ollama(messages, question, model, save_history, history_file, conversation_history, force_reprocess)`
- Main function for analyzing messages with LLM
- Handles conversation chunking and token optimization
- Supports caching and conversation history

##### `analyze_individual_conversation(conv_id, csv_file, question, model, save_history, history_file, conversation_history, force_reprocess)`
- Analyzes a single conversation by ID
- Useful for deep-diving into specific conversations

##### `group_messages_by_conversation(messages)`
- Groups messages by their conversation_id
- Returns dictionary mapping conversation IDs to message lists

##### `filter_conversations_by_criteria(conversations, min_messages, max_messages, include_ids, exclude_ids)`
- Filters conversations based on various criteria
- Useful for focusing analysis on specific conversation types

##### `create_conversation_chunks(conversations, question, max_context_tokens)`
- Creates token-optimized chunks for LLM processing
- Preserves conversation boundaries while respecting token limits

### Configuration Options

#### Adaptive Sampling
The tool automatically determines optimal sample sizes based on dataset size and analysis type:
- **General analysis**: 1,000+ messages minimum
- **Conversation quality**: 2,000+ messages minimum  
- **Momentum analysis**: 500+ messages minimum
- **Context analysis**: 300+ messages minimum

#### Window Sizes
Context window sizes adapt based on dataset size:
- **Small datasets (<1,000)**: Base window sizes (3-5 messages)
- **Medium datasets (1,000-10,000)**: Base + 1
- **Large datasets (10,000-50,000)**: Base + 2  
- **Very large datasets (50,000+)**: Base + 3 (max 12)

#### Confidence Thresholds
Decision confidence levels for conversation breaks:
- **High confidence (0.9-1.0)**: Time-based, conversation starters/enders
- **Medium confidence (0.7-0.8)**: Topic changes, momentum shifts
- **Lower confidence (0.5-0.6)**: Emotional shifts, activity transitions

## Data Flow

1. **Input**: iMessage HTML export (`data/message.html`)
2. **Processing**: `process.py` extracts and groups messages
3. **Output**: Structured CSV (`data/messages.csv`) with conversation IDs
4. **Analysis**: `ask_messages.py` uses the CSV for AI-powered insights
5. **Results**: Cached analyses and saved conversation histories

## Output Formats

### CSV File Structure (data/messages.csv)

The processed CSV file contains the following columns:

- **date**: Timestamp in `YYYY-MM-DD HH:MM:SS` format
- **sender**: Either "Me" or "Them" (standardized)
- **message**: The message content
- **conversation_id**: Unique identifier for each conversation

### Cached Analysis Files (cache/)

- JSON files containing analysis results
- Keyed by conversation ID, question, and message hash
- Includes response text, metadata, and timestamps

### Conversation History Files (conversations/)

- JSON files tracking analysis sessions
- Contains questions, responses, and conversation metadata
- Automatically saved with timestamps or custom filenames

## Sample Analytics Output

```
═══════════════════════════════════ CONVERSATION ANALYSIS ════════════════════════════════════
                                Analyzing conversation patterns...

✓ Assigned 1,234 conversations in 2.3s

────────────────────────────────── Decision Statistics ──────────────────────────────────
  Time Based: 856 (69.4%)
  Topic Change: 187 (15.2%)
  Starter Based: 98 (7.9%)
  Momentum Based: 93 (7.5%)

────────────────────────────────── Message Statistics ──────────────────────────────────
  Average messages per conversation: 8.4
  Shortest conversation: 1 messages
  Longest conversation: 342 messages
  Single message conversations: 234
  Short conversations (2-10 msgs): 789
  Long conversations (11+ msgs): 211
```

## Performance Considerations

- **Large Files**: The tool efficiently handles multi-gigabyte HTML files using streaming
- **Memory Usage**: Memory-efficient processing keeps RAM usage reasonable even for large datasets
- **Processing Speed**: Optimized regex parsing and analysis algorithms for fast processing
- **Sample Analysis**: Automatic sampling for very large datasets to maintain reasonable processing times

## Troubleshooting

### Common Issues

#### Processing Issues (process.py)
1. **"No such file or directory" error**: Ensure `data/message.html` exists
2. **Memory errors**: For very large files, the tool uses chunked processing automatically
3. **Encoding issues**: The tool handles UTF-8 encoding for international characters and emojis
4. **Empty output**: Check that your HTML file follows the expected iMessage export format

#### Analysis Issues (ask_messages.py)
1. **"Ollama not running" error**: Start Ollama with `ollama serve`
2. **"Model not found" error**: Install DeepSeek with `ollama pull deepseek-r1:14b`
3. **"No conversation_id" error**: Run `process.py` first to process your messages
4. **Token limit exceeded**: The tool automatically chunks large datasets, but very long single conversations may need special handling
5. **Cache issues**: Use `--clear-cache` to reset cached results if experiencing problems

### Performance Tips

- For very large datasets (100k+ messages), consider using the quality analysis on samples first
- The tool automatically optimizes processing based on dataset size
- Progress indicators help track processing on large files

## Dependencies

Key dependencies (see `requirements.txt` for complete list):
- **beautifulsoup4**: HTML parsing (backup method)
- **tiktoken**: Token counting for LLM context management
- **requests**: API communication with Ollama
- **Standard library**: Primary processing uses built-in Python modules for speed

### External Requirements
- **Ollama**: Local LLM server (https://ollama.ai)
- **DeepSeek-R1 Model**: Run `ollama pull deepseek-r1:14b` after installing Ollama

## Technical Details

### Conversation Detection Algorithm

The conversation grouping uses a multi-layered approach:

1. **Primary Breaks**: Major time gaps, strong conversation indicators
2. **Secondary Breaks**: Momentum changes, topic shifts, pattern disruptions  
3. **Contextual Analysis**: Surrounding message analysis for continuity
4. **Preservation Rules**: Protects rapid exchanges and high-engagement conversations

### Statistical Significance

- Uses proper sample size calculations with confidence intervals
- Applies finite population corrections for small datasets
- Validates sample adequacy for statistical significance

### Error Handling

- Graceful handling of malformed HTML
- Robust timestamp parsing with fallback methods
- Comprehensive error reporting with clear messages

## Typical Workflow

1. **Export Messages**: Use [imessage-exporter](https://github.com/ReagentX/imessage-exporter) to export your iMessage conversations to HTML format
2. **Prepare Data**: Place the HTML export in `data/message.html`
3. **Process Data**: Run `process.py` to convert HTML to structured CSV with conversation IDs
4. **Explore Conversations**: Use `--list-conversations` to see available conversations
5. **Analyze**: Ask questions about specific conversations, time periods, or the entire dataset
6. **Iterate**: Continue analysis sessions with follow-up questions using conversation history
7. **Cache Results**: Reuse cached analyses for faster repeated queries

### Data Export Options

While we recommend [imessage-exporter](https://github.com/ReagentX/imessage-exporter) for its reliability and features, you can use any tool that exports iMessages to HTML format with the following structure:
- Message containers with class "message"
- Timestamps in a "timestamp" span
- Sender names in a "sender" span
- Message content in a "bubble" span

Common export tools:
- **imessage-exporter** (recommended): Fast, reliable, multiple format support
- **iMessage Export**: GUI tool for macOS
- **PhoneView**: Commercial tool with export features
- Manual export from Messages app (limited functionality)

## Use Cases

- **Relationship Analysis**: Understand how relationships evolved over time
- **Topic Discovery**: Identify recurring themes and conversation patterns
- **Memory Lane**: Revisit specific conversations or time periods
- **Communication Patterns**: Analyze response times, conversation lengths, and engagement
- **Emotional Journey**: Track emotional tones and relationship dynamics
- **Content Search**: Find specific topics or events in your message history

## Contributing

When contributing to this project, please:
1. Follow the existing code style and formatting
2. Add appropriate error handling and logging
3. Update documentation for new features
4. Test with various HTML export formats

## License

This project is open source. Please see the license file for details.
