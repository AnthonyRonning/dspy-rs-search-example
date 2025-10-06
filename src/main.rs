use anyhow::Result;
use dspy_rs::*;
use std::io::{self, Write};
use std::env;
use std::sync::Arc;
use tokio::sync::Mutex;

// ============================================================================
// TOOLS - Structured programs that do specific work
// ============================================================================

// Mock search function - replace with real search API
async fn search_web(_query: &str) -> String {
    "Trump is currently the president in 2025".to_string()
}

/// SearchTool - Performs web search and returns structured results
#[Signature]
struct SearchQuery {
    /// Extract the main search query from the user's question.
    /// Return only the search terms, nothing else.

    #[input]
    pub user_question: String,

    #[output]
    pub search_query: String,
}

pub struct SearchTool {
    query_extractor: Predict,
    lm: Arc<Mutex<LM>>,
}

impl SearchTool {
    fn new(lm: Arc<Mutex<LM>>) -> Self {
        Self {
            query_extractor: Predict::new(SearchQuery::new()),
            lm,
        }
    }

    async fn search(&self, user_question: &str) -> Result<(String, String)> {
        // Extract search query
        let example = example! {
            "user_question": "input" => user_question,
        };

        let query_result = self.query_extractor.forward_with_config(example, Arc::clone(&self.lm)).await?;
        let query = query_result.get("search_query", None).as_str().unwrap().to_string();

        // Perform search
        let results = search_web(&query).await;

        Ok((query, results))
    }
}

// ============================================================================
// CLASSIFIER - Lightweight, fast intent detection
// ============================================================================

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

pub struct IntentClassifier {
    classifier: Predict,
    lm: Arc<Mutex<LM>>,
}

impl IntentClassifier {
    fn new(lm: Arc<Mutex<LM>>) -> Self {
        Self {
            classifier: Predict::new(IntentClassification::new()),
            lm,
        }
    }

    async fn classify(&self, message: &str) -> Result<String> {
        let example = example! {
            "user_message": "input" => message,
        };

        let result = self.classifier.forward_with_config(example, Arc::clone(&self.lm)).await?;
        let intent = result.get("intent", None).as_str().unwrap().to_lowercase();

        // Normalize to expected values
        if intent.contains("search") {
            Ok("search".to_string())
        } else {
            Ok("chat".to_string())
        }
    }
}

// ============================================================================
// PERSONALITY - Natural conversational response
// ============================================================================

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

pub struct PersonalityChat {
    responder: Predict,
    lm: Arc<Mutex<LM>>,
}

impl PersonalityChat {
    fn new(lm: Arc<Mutex<LM>>) -> Self {
        Self {
            responder: Predict::new(PersonalityResponse::new()),
            lm,
        }
    }

    async fn respond(
        &self,
        user_message: &str,
        conversation_history: &str,
        search_results: Option<&str>,
    ) -> Result<String> {
        let example = example! {
            "conversation_history": "input" => conversation_history,
            "user_message": "input" => user_message,
            "search_results": "input" => search_results.unwrap_or(""),
        };

        let result = self.responder.forward_with_config(example, Arc::clone(&self.lm)).await?;
        Ok(result.get("response", None).as_str().unwrap().to_string())
    }
}

// ============================================================================
// ORCHESTRATOR - Coordinates classifier → tools → personality
// ============================================================================

pub struct ConversationalAgent {
    classifier: IntentClassifier,
    search_tool: SearchTool,
    personality: PersonalityChat,
}

impl ConversationalAgent {
    fn new(classifier_lm: Arc<Mutex<LM>>, personality_lm: Arc<Mutex<LM>>) -> Self {
        Self {
            classifier: IntentClassifier::new(Arc::clone(&classifier_lm)),
            search_tool: SearchTool::new(classifier_lm),  // Reuse classifier LM for tools
            personality: PersonalityChat::new(personality_lm),
        }
    }
}

impl Module for ConversationalAgent {
    async fn forward(&self, inputs: Example) -> Result<Prediction> {
        let user_message = inputs.data.get("user_message").unwrap().to_string();
        let conversation_history = inputs.data.get("conversation_history")
            .map(|v| v.to_string())
            .unwrap_or_else(|| String::new());

        // Step 1: Classify intent (using fast model)
        println!("🔍 Classifying intent...");
        let intent = self.classifier.classify(&user_message).await?;

        // Step 2: Execute appropriate tool if needed
        let search_results = if intent == "search" {
            match self.search_tool.search(&user_message).await {
                Ok((query, results)) => {
                    println!("📋 Intent: search(\"{}\")\n", query);
                    println!("🌐 Performing search...");
                    println!("✅ Search complete\n");
                    Some(results)
                }
                Err(e) => {
                    println!("📋 Intent: {}\n", intent);
                    println!("⚠️  Search failed: {}\n", e);
                    None
                }
            }
        } else {
            println!("📋 Intent: {}\n", intent);
            None
        };

        // Step 3: Generate natural response with personality module
        println!("💭 Generating response...");
        let response = self.personality.respond(
            &user_message,
            &conversation_history,
            search_results.as_deref(),
        ).await?;

        Ok(prediction! {
            "response" => response,
        })
    }
}

// ============================================================================
// CLI
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    let api_key = std::env::var("OPENAI_API_KEY")?;

    // Classifier LM: Fast, cheap model for intent classification
    let classifier_lm = Arc::new(Mutex::new(
        LM::builder()
            .api_key(api_key.clone().into())
            .config(
                LMConfig::builder()
                    .model("gpt-4o-mini".to_string())
                    .temperature(0.0)  // Deterministic classification
                    .build(),
            )
            .build()
    ));

    // Personality LM: Better model for natural conversation
    let personality_model = env::var("PERSONALITY_MODEL")
        .unwrap_or_else(|_| "gpt-4o".to_string());

    let personality_lm = Arc::new(Mutex::new(
        LM::builder()
            .api_key(api_key.into())
            .config(
                LMConfig::builder()
                    .model(personality_model)
                    .temperature(0.7)  // Natural, varied responses
                    .build(),
            )
            .build()
    ));

    // Still need to configure global settings (for any modules that use default forward())
    configure(
        LM::builder()
            .api_key(std::env::var("OPENAI_API_KEY")?.into())
            .build(),
        ChatAdapter
    );

    // Create the conversational agent with separate LMs
    let agent = ConversationalAgent::new(classifier_lm, personality_lm);

    // Parse command-line arguments
    let args: Vec<String> = env::args().collect();

    // Check for -p flag (one-shot mode)
    if args.len() >= 3 && args[1] == "-p" {
        let question = &args[2];

        let example = example! {
            "conversation_history": "input" => "",
            "user_message": "input" => question,
        };

        let result = agent.forward(example).await?;
        println!("\n{}", result.get("response", None).as_str().unwrap());

        return Ok(());
    }

    // Interactive mode
    println!("🤖 Conversational Agent with Classifier Architecture");
    println!("💡 Using fast classifier → tools → personality flow");
    println!("Type your messages below (Ctrl+C to exit)\n");
    println!("{}", "=".repeat(60));

    // Maintain conversation history
    let mut conversation_history = Vec::new();

    loop {
        print!("\n💬 You: ");
        io::stdout().flush()?;

        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(0) => break, // EOF
            Ok(_) => {
                let message = input.trim();

                if message.is_empty() {
                    continue;
                }

                if message.eq_ignore_ascii_case("exit") || message.eq_ignore_ascii_case("quit") {
                    println!("\n👋 Goodbye!");
                    break;
                }

                // Format history
                let history_str = if conversation_history.is_empty() {
                    String::new()
                } else {
                    conversation_history.join("\n")
                };

                let example = example! {
                    "conversation_history": "input" => history_str,
                    "user_message": "input" => message,
                };

                match agent.forward(example).await {
                    Ok(result) => {
                        let response = result.get("response", None).as_str().unwrap().to_string();
                        println!("\n🤖 Agent: {}\n", response);
                        println!("{}", "=".repeat(60));

                        // Add to history
                        conversation_history.push(format!("User: {}", message));
                        conversation_history.push(format!("Assistant: {}", response));
                    }
                    Err(e) => {
                        eprintln!("\n❌ Error: {}\n", e);
                        println!("{}", "=".repeat(60));
                    }
                }
            }
            Err(e) => {
                eprintln!("\n❌ Error reading input: {}", e);
                break;
            }
        }
    }

    Ok(())
}
