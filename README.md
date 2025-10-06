# DSPy Classifier-First Agent (Rust)

> **A proof-of-concept conversational AI architecture using DSPy Rust that separates intent classification, tool execution, and personality**

This MVP demonstrates how to build conversational agents with a **classifier-first architecture** instead of the traditional ReAct pattern. By using a lightweight classifier upfront, we keep the personality module clean and natural while efficiently routing to structured tools.

## What This Demonstrates

✅ **Classifier-first architecture** - Route intents before engaging personality
✅ **Separation of concerns** - Intent detection, tools, and personality are independent modules
✅ **Cost optimization** - Use fast/cheap models (gpt-4o-mini) for classification
✅ **Clean personality** - No bloated context with tool instructions
✅ **DSPy modularity** - Each component is a composable DSPy module
✅ **Extensibility** - Easy to add new tools without touching personality

## Overview

This project implements a **classifier-first architecture** for building conversational AI agents using DSPy Rust. The design separates concerns between intent detection, structured tool execution, and natural personality responses.

## Architecture Diagram

```
┌─────────────────┐
│   User Input    │
└────────┬────────┘
         │
         ▼
┌──────────────────────────────────┐
│   IntentClassifier               │  ← gpt-4o-mini (fast & cheap)
│   Returns: "search" | "chat"     │    ~$0.00015/1K tokens
└────────┬─────────────────────────┘    Temperature: 0.0 (deterministic)
         │
         ├──→ "search" ──→ ┌─────────────────┐
         │                 │   SearchTool    │  ← gpt-4o-mini (reused)
         │                 │   (web search)  │    Fast structured program
         │                 └────────┬────────┘
         │                          │
         └──→ "chat" ───────────┐   │
                                │   │
                                ▼   ▼
                         ┌──────────────────────┐
                         │  PersonalityChat     │  ← gpt-4o (default)
                         │  (natural, friendly) │    Better conversation
                         └──────────┬───────────┘    Temperature: 0.7 (creative)
                                    │
                                    ▼
                            User Response
```

## Signatures & Modules

This implementation uses **3 DSPy Signatures** and **4 Modules**:

### DSPy Signatures

**1. IntentClassification** - Classify user intent
```rust
#[Signature]
struct IntentClassification {
    /// Classify the user's intent. Return ONLY one of these exact values:
    /// - "search" if the user needs current information, facts, or web search
    /// - "chat" if the user wants casual conversation, greetings, or general discussion

    #[input]
    pub user_message: String,

    #[output]
    pub intent: String,
}
```

**2. SearchQuery** - Extract search terms from user questions
```rust
#[Signature]
struct SearchQuery {
    /// Extract the main search query from the user's question.
    /// Return only the search terms, nothing else.

    #[input]
    pub user_question: String,

    #[output]
    pub search_query: String,
}
```

**3. PersonalityResponse** - Generate natural conversational responses
```rust
#[Signature]
struct PersonalityResponse {
    /// You are a friendly, helpful AI assistant. Respond naturally and conversationally.
    /// If search results are provided, use them to answer the question accurately.
    /// If no search results, just have a natural conversation.
    /// Consider conversation history for context.

    #[input]
    pub conversation_history: String,

    #[input]
    pub user_message: String,

    #[input]
    pub search_results: String,

    #[output]
    pub response: String,
}
```

### Modules

**1. IntentClassifier** - Uses `IntentClassification` signature with gpt-4o-mini

**2. SearchTool** - Uses `SearchQuery` signature with gpt-4o-mini + mock search function

**3. PersonalityChat** - Uses `PersonalityResponse` signature with gpt-4o

**4. ConversationalAgent** - Orchestrator that composes all modules together

## Why This Architecture?

### Key Benefits

1. **Performance**: Lightweight classifier (gpt-4o-mini) handles routing - fast and cheap
2. **Clean Separation**: Personality module isn't bloated with routing logic or tool instructions
3. **Flexibility**: Easy to swap LLMs for different components
4. **Extensibility**: Simple to add new tools (calculator, code executor, etc.)
5. **Cost Efficiency**: Use expensive models only for personality, cheap models for classification

### Comparison to ReAct Pattern

**ReAct (Single Agent):**
- One agent handles both personality AND tool decisions
- Heavier prompts include all tool documentation
- More tokens per request
- Agent must balance being conversational + tool-aware

**Classifier-First (This Implementation):**
- Classifier handles intent (dedicated, optimized)
- Personality module stays natural and clean
- Fewer tokens for personality responses
- Each component has single responsibility

## Components

### 1. IntentClassifier (`src/main.rs:76-104`)

**Purpose**: Fast, lightweight intent detection

**Model**: `gpt-4o-mini` (temperature 0.0 for deterministic classification)

**Returns**: `"search"` | `"chat"`

**Why separate?**
- Fast response times (~2-10x faster than gpt-4o)
- Cost-effective (~15x cheaper - classifier runs on every message!)
- Can be optimized independently with DSPy optimizers
- Easy to add more intents later

### 2. SearchTool (`src/main.rs:30-57`)

**Purpose**: Structured search execution

**Model**: `gpt-4o-mini` (reused from classifier - cheap for extraction)

**Components**:
- Query extraction (DSPy signature)
- Web search execution
- Result formatting

**Why as a tool?**
- Encapsulates search logic
- Reusable across different agents
- Can be tested/optimized independently
- Easy to swap search providers
- Uses cheap model for structured tasks

### 3. PersonalityChat (`src/main.rs:130-158`)

**Purpose**: Natural conversational responses

**Model**: `gpt-4o` (default, configurable via `PERSONALITY_MODEL` env var)

**Input**:
- User message
- Conversation history
- Optional search results

**Why separate?**
- Keeps conversational tone natural
- Uses better model for nuanced conversation
- Minimal context bloat
- User can configure their preferred LLM
- No tool documentation in prompts

### 4. ConversationalAgent (Orchestrator) (`src/main.rs:164-220`)

**Purpose**: Coordinates the full pipeline

**Flow**:
1. Classify intent
2. Execute tools if needed
3. Generate natural response with personality module

## Extending the Architecture

### Adding a New Tool

1. **Create Tool Module**:
```rust
pub struct CalculatorTool {
    calculator: Predict,
    lm: Arc<Mutex<LM>>,
}

impl CalculatorTool {
    fn new(lm: Arc<Mutex<LM>>) -> Self {
        Self {
            calculator: Predict::new(CalculatorSignature::new()),
            lm,
        }
    }

    async fn calculate(&self, expression: &str) -> Result<String> {
        // Implementation
    }
}
```

2. **Update Classifier**:
```rust
#[Signature]
struct IntentClassification {
    /// Return: "search" | "chat" | "calculate"
    // ...
}
```

3. **Add to Orchestrator**:
```rust
pub struct ConversationalAgent {
    classifier: IntentClassifier,
    search_tool: SearchTool,
    calculator_tool: CalculatorTool,  // New
    personality: PersonalityChat,
}
```

4. **Handle in Forward**:
```rust
let tool_results = match intent.as_str() {
    "search" => Some(self.search_tool.search(&user_message).await?),
    "calculate" => Some(self.calculator_tool.calculate(&user_message).await?),
    _ => None,
};
```

### Using Different LLMs per Component

✅ **This is already implemented!** Each component uses its own LM:

**Current Configuration** (`src/main.rs:229-269`):
```rust
// Classifier & Tools: gpt-4o-mini (fast & cheap)
let classifier_lm = Arc::new(Mutex::new(
    LM::builder()
        .api_key(api_key.clone().into())
        .config(
            LMConfig::builder()
                .model("gpt-4o-mini".to_string())
                .temperature(0.0)  // Deterministic
                .build(),
        )
        .build()
));

// Personality: gpt-4o (better conversation)
let personality_lm = Arc::new(Mutex::new(
    LM::builder()
        .api_key(api_key.into())
        .config(
            LMConfig::builder()
                .model("gpt-4o".to_string())  // Or set PERSONALITY_MODEL env var
                .temperature(0.7)  // Natural variance
                .build(),
        )
        .build()
));

// Each module gets its appropriate LM
let agent = ConversationalAgent::new(classifier_lm, personality_lm);
```

**How It Works**:
- Uses DSPy's `forward_with_config()` method to pass custom LMs
- Classifier and tools share the cheap `gpt-4o-mini` model
- Personality gets the better `gpt-4o` model
- Each module stores its LM and uses `forward_with_config()` instead of `forward()`

**Customization**:
```bash
# Change personality model
export PERSONALITY_MODEL="gpt-4o-mini"  # Use cheaper model
export PERSONALITY_MODEL="gpt-4"        # Use older model
export PERSONALITY_MODEL="gpt-4o"       # Default
```

## DSPy Philosophy Alignment

This architecture follows DSPy's core principles:

1. **Modular**: Each component is a reusable module
2. **Composable**: Modules are composed in the orchestrator
3. **Optimizable**: Each module can be optimized independently with DSPy optimizers
4. **Declarative**: Uses signatures to define input/output contracts

From DSPy docs:
> "DSPy separates these concerns and automates the lower-level ones until you need to consider them."

This architecture separates:
- **Routing** (classifier)
- **Tool execution** (search, future tools)
- **Presentation** (personality)

Each can be optimized, swapped, or extended independently.

## About This POC

This is a proof-of-concept demonstrating an alternative to the standard ReAct agent pattern. The goal was to answer:

**"How can we build conversational agents where the personality stays natural and clean, while still having access to structured tools?"**

The answer: **Use a lightweight classifier to route intents upfront, keeping the personality module focused solely on natural conversation.**

This approach is particularly useful when:
- You want different LLMs for different tasks (cheap for routing, premium for personality)
- You need predictable, optimizable intent classification
- You want to add tools without bloating conversational prompts
- You value separation of concerns and modularity

## Project Structure

```
dspy-search/
├── src/
│   └── main.rs              # All components (classifier, tools, personality, orchestrator)
├── DSRs/                    # DSPy Rust submodule
├── Cargo.toml
└── README.md
```

## Future Enhancements

- [ ] Add more tools (calculator, code executor, etc.)
- [ ] Add DSPy optimizers (COPRO) to improve classifier
- [ ] Add conversation memory/RAG
- [ ] Add tool chaining (use multiple tools in sequence)
- [ ] Add streaming responses
- [ ] Add evaluation metrics

## Quick Start

### Prerequisites

- Rust 1.70+ (`cargo --version`)
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd dspy-search

# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."

# Optional: Choose personality model (defaults to gpt-4o)
export PERSONALITY_MODEL="gpt-4o"  # Better conversation
# export PERSONALITY_MODEL="gpt-4o-mini"  # Faster/cheaper

# Build and run
cargo build
cargo run
```

### Usage

```bash
# Interactive mode
cargo run

# One-shot queries
cargo run -- -p "hello there!"              # Natural chat
cargo run -- -p "who is the president?"     # Triggers search
```

### Testing

```bash
# Build
cargo build

# Run tests (when added)
cargo test

# Test classification
cargo run -- -p "hello"                # Should classify as "chat"
cargo run -- -p "who is president?"    # Should classify as "search"
```

## References

- [DSPy Documentation](https://dspy.ai)
- [DSRs (DSPy Rust) Repo](https://github.com/krypticmouse/dsrs)
- [DSPy Programming Overview](https://dspy.ai/learn/programming/overview/)
- [DSPy Conversation History](https://dspy.ai/tutorials/conversation_history/)

---

**Built with [DSPy Rust (DSRs)](https://github.com/krypticmouse/dsrs)** 🦀
