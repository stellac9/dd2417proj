import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_summaries(input_text, model, tokenizer, max_length=300, num_return_sequences=1):
    inputs = tokenizer.encode(input_text, add_special_tokens=True, return_tensors="pt")
    
    # Generate the summaries
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=num_return_sequences)
    
    generated_summaries = []
    for output in outputs:
        summary = tokenizer.decode(output, skip_special_tokens=True)
        summary = summary[len(input_text):]
        print(summary)

        #last_period_index = summary.rfind(".")
        #if last_period_index != -1:
            #summary = summary[:last_period_index + 1] 

        generated_summaries.append(summary)
    
    return generated_summaries

def main():
    # Load the fine-tuned model
    model = GPT2LMHeadModel.from_pretrained("fine-tuned-model")
    model.eval()
    
    # Set up the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Example usage
    input_text = "Before the advent of smartphones with which chatting and text messaging can be done for free over the Internet, there was Chikka Text Messenger, or simply Chikka. Chikka is an Internet-based messaging application that offers users the ability to send SMS or text messages for free. It is the first instant messaging application on the computer that allowed for SMS exchanges with mobile phones. Established in the Philippines, it has continued presence and has since expanded its operations internationally. It is also now available across different platforms and devices. On the Android, even with the presence of other messaging apps, Chikka still holds a niche since it allows free text message exchanges even with non-Chikka, non-Android and non-smartphone users."
    summaries = generate_summaries(input_text, model, tokenizer)


    # Print the generated summaries
    """ for i, summary in enumerate(summaries):
        print(f"Generated Summary {i+1}: {summary}") """

if __name__ == '__main__':
    main()
