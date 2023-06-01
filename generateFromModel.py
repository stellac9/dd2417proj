import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

def generate_summaries(input_text, model, tokenizer, max_length=100, num_return_sequences=1):
    inputs = tokenizer.encode(input_text, add_special_tokens=True, return_tensors="pt")   
    # Generate the summaries
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=num_return_sequences)

    #print(tokenizer.decode(outputs))
    generated_summaries = []
    for output in outputs:
        summary = tokenizer.decode(output, skip_special_tokens=True)
        
        summary = summary[len(input_text):]
        #print(summary)

        #last_period_index = summary.rfind(".")
        #if last_period_index != -1:
            #summary = summary[:last_period_index + 1] 

        generated_summaries.append(summary)
    
    return generated_summaries

def main():
    # Load the fine-tuned model
    model = GPT2LMHeadModel.from_pretrained("fine-tuned-model-smaller")
    model.eval()
    
    # Set up the tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # Example usage
    # Secure the bat.
    input_text = " Make sure the bat is in a secure position where it can’t move. You can place the bat between your legs or have a friend hold the bat for you.Make sure you don’t tighten things too much if you secure your bat using a clamp."
    summaries = generate_summaries(input_text, model, tokenizer)
    print(summaries[0])

    # Print the generated summaries
"""     for i, summary in enumerate(summaries):
        print(f"Generated Summary {i+1}: {summary}") """

if __name__ == '__main__':
    main()
