import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from rouge_score import rouge_scorer

def generate_summaries(input_text, model, tokenizer, max_length=100, num_return_sequences=1):
    inputs = tokenizer.encode(input_text, add_special_tokens=True, return_tensors="pt")   
    # Generate the summaries
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=num_return_sequences)

    generated_summaries = []
    for output in outputs:
        summary = tokenizer.decode(output, skip_special_tokens=True)
        summary = summary[len(input_text):]
        generated_summaries.append(summary)
    
    return generated_summaries

def evaluate_summaries(references, summaries):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    valid_summaries = [s for s in summaries if s is not None and s != '']
    # rouge1 - unigram based scoring, # rouge 2 - bigram based scoring, rougeL - longest common subsequence based scoring
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for reference in references:
        for summary in valid_summaries:
            scores = scorer.score(reference, summary)
            for metric, score in scores.items():
                rouge_scores[metric].append(score.fmeasure)
    
    avg_scores = {metric: sum(scores) / len(scores) for metric, scores in rouge_scores.items()}
    return avg_scores

def main():
    # Load the fine-tuned model
    model = GPT2LMHeadModel.from_pretrained("fine-tuned-model-smaller")
    model.eval()
    
    # Set up the tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # Example usage
    input_text = " Make sure the bat is in a secure position where it can’t move. You can place the bat between your legs or have a friend hold the bat for you.Make sure you don’t tighten things too much if you secure your bat using a clamp."
    references = ["Secure the bat."]
    summaries = generate_summaries(input_text, model, tokenizer)
    print(summaries)
    
    # Evaluate the generated summaries
    scores = evaluate_summaries(references, summaries)
    print("ROUGE scores:")
    for metric, score in scores.items():
        print(f"{metric}: {score}")


if __name__ == '__main__':
    main()
