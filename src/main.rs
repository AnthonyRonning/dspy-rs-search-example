use anyhow::Result;
use dspy_rs::*;

// Mock search function - always returns the same result
async fn search_web(_query: &str) -> String {
    "Trump is currently the president in 2025".to_string()
}

#[Signature]
struct InitialAnswer {
    /// Answer the question if you know the answer with certainty.
    /// If you don't know or need current information, respond with exactly "NEED_SEARCH".
    /// Provide answers directly without quotes.

    #[input]
    pub question: String,

    #[output]
    pub response: String,
}

#[Signature]
struct AnswerWithContext {
    /// Answer the question using the provided search results as context.

    #[input]
    pub question: String,

    #[input]
    pub search_results: String,

    #[output]
    pub answer: String,
}

struct SearchAgent {
    initial_answerer: Predict,
    context_answerer: Predict,
}

impl SearchAgent {
    fn new() -> Self {
        Self {
            initial_answerer: Predict::new(InitialAnswer::new()),
            context_answerer: Predict::new(AnswerWithContext::new()),
        }
    }
}

impl Module for SearchAgent {
    async fn forward(&self, inputs: Example) -> Result<Prediction> {
        let question = inputs.data.get("question").unwrap().clone();

        // Step 1: Try to answer directly
        println!("🤔 Attempting to answer directly...\n");
        let initial = self.initial_answerer.forward(inputs.clone()).await?;
        let response = initial.data.get("response").unwrap().to_string();

        // Step 2: Check if search is needed
        if response.contains("NEED_SEARCH") {
            println!("🔍 Need to search for information...\n");

            // Call search tool
            let search_results = search_web(&question.to_string()).await;
            println!("📄 Search results: {}\n", search_results);

            // Step 3: Answer with context
            let with_context = example! {
                "question": "input" => question,
                "search_results": "input" => search_results,
            };

            println!("💡 Generating answer with search context...\n");
            self.context_answerer.forward(with_context).await
        } else {
            println!("✅ Answered directly without search\n");
            // Return the initial prediction but rename the field from "response" to "answer" for consistency
            // Strip surrounding quotes if present
            let cleaned_response = response.trim_matches(|c| c == '"' || c == '\'');
            Ok(prediction! {
                "answer" => cleaned_response,
            })
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let lm = LM::builder()
        .api_key(std::env::var("OPENAI_API_KEY")?.into())
        .config(
            LMConfig::builder()
                .model("gpt-4o-mini".to_string())
                .temperature(0.5)
                .build(),
        )
        .build();

    configure(lm, ChatAdapter);

    // Create the search agent module
    let agent = SearchAgent::new();

    // Test with a question that requires search (current information)
    println!("{}", "=".repeat(60));
    println!("Question 1: Who is the current US president?");
    println!("{}", "=".repeat(60));

    let example1 = example! {
        "question": "input" => "Who is the current US president?",
    };

    let result1 = agent.forward(example1).await?;
    println!(
        "Final Answer: {}\n",
        result1.get("answer", None).as_str().unwrap()
    );

    // Test with a question that doesn't require search (general knowledge)
    println!("{}", "=".repeat(60));
    println!("Question 2: What is 2+2?");
    println!("{}", "=".repeat(60));

    let example2 = example! {
        "question": "input" => "What is 2+2?",
    };

    let result2 = agent.forward(example2).await?;
    println!(
        "Final Answer: {}",
        result2.get("answer", None).as_str().unwrap()
    );

    Ok(())
}
