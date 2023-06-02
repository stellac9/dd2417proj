import csv
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from rouge_score import rouge_scorer

def generate_summaries(input_text, model, tokenizer, max_length=300, num_return_sequences=1):
    inputs = tokenizer.encode(input_text, add_special_tokens=True, return_tensors="pt")   
    # Generate the summaries
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=num_return_sequences)

    generated_summaries = []
    for output in outputs:
        summary = tokenizer.decode(output, skip_special_tokens=True)
        summary = summary[len(input_text):]
        summary = summary.split('. ')[0] + '.'
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
    # Load the fine-tuned model -- change model name if needed
    model = GPT2LMHeadModel.from_pretrained("fine-tuned-model-amaz200k")
    model.eval()
    
    # Set up the tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # Read input texts and references from CSV file
    input_texts = []
    references = []
    # change file to read from if needed
    with open("AmazonFineFoods/reviewsTest.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # change indices here to the correct columns for the CSV being used
            input_texts.append(row[9])
            references.append(row[8])
    
    # Generate and evaluate summaries for each input text
    for input_text, reference in zip(input_texts, references):
        summaries = generate_summaries(input_text, model, tokenizer)
        print("Input Text:", input_text)
        print("Generated Summary:", summaries[0])
        print("Reference:", reference)
        
        # Evaluate the generated summary
        scores = evaluate_summaries([reference], summaries)
        print("ROUGE scores:")
        for metric, score in scores.items():
            print(f"{metric}: {score}")
        print()
        

if __name__ == '__main__':
    main()
