use anyhow::Result;
use dspy_rs::*;

#[Signature]
struct SentimentAnalyzer {
    /// Predict the sentiment of the given text 'Positive', 'Negative', or 'Neutral'.

    #[input]
    pub text: String,

    #[output]
    pub sentiment: String,
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

    // Create a predictor
    let predictor = Predict::new(SentimentAnalyzer::new());

    // Prepare input
    let example = example! {
        "text": "input" => "Acme is a great company with excellent customer service.",
    };

    // Execute prediction
    let result = predictor.forward(example).await?;

    println!("Answer: {}", result.get("sentiment", None));

    Ok(())
}
