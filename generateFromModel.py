import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_summaries(input_text, model, tokenizer, max_length=100, num_return_sequences=1):
    inputs = tokenizer.encode(input_text, add_special_tokens=True, return_tensors="pt")
    
    # Generate the summaries
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=num_return_sequences)
    
    generated_summaries = []
    for output in outputs:
        summary = tokenizer.decode(output, skip_special_tokens=True)
        generated_summaries.append(summary)
    
    return generated_summaries

def main():
    # Load the fine-tuned model
    model = GPT2LMHeadModel.from_pretrained("fine-tuned-model")
    model.eval()
    
    # Set up the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Example usage
    input_text = "Your input text goes here."
    summaries = generate_summaries(input_text, model, tokenizer)
    
    # Print the generated summaries
    for i, summary in enumerate(summaries):
        print(f"Generated Summary {i+1}: {summary}")

if __name__ == '__main__':
    main()
