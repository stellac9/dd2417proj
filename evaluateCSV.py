import csv
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from rouge_score import rouge_scorer

def generate_summaries(input_text, model, tokenizer, max_length=300, num_return_sequences=1):
    """
    function that takes in the input_text, the model that was previously trained and the GPT2 tokenizer
    and returns the generated summaries by decoding the tokenizer
    """

    # encode the input text using the tokenizer.encode method and store the encodings in inputs
    inputs = tokenizer.encode(input_text, add_special_tokens=True, return_tensors="pt")   
    # generate the summaries by passing the encoded inputs to the model.generate method and pass the result to outputs
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=num_return_sequences, pad_token_id=tokenizer.eos_token_id)

    # create an empty list of generated summaries
    generated_summaries = []

    # for each output
    for output in outputs:
        # decode the outputs that were generated and pass the result to summary
        summary = tokenizer.decode(output, skip_special_tokens=True)
        # remove the original input text from summary and allocate '.' or '!' as places for the summary to stop
        summary = summary[len(input_text):]
        stop_indices = [summary.find('.'), summary.find('!')]
        # if one of the above symbols occurs then set the summary to the text up to that point.
        valid_stops = [i for i in stop_indices if i != -1]
        if valid_stops:
            summary = summary[:min(valid_stops) + 1]
        # append the summary to the generated_summaries list
        generated_summaries.append(summary)

    return generated_summaries

def evaluate_summaries(references, summaries):
    """
    function to evaluate the accuracy of the summaries against some reference summaries using ROUGE.
    function takes in the references and the generated summaries and returns the average scores for each ROUGE type
    """

    # call RougeScorer and use rouge1, rouge2 and rougeL
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    # make sure the summary is valid
    valid_summaries = [s for s in summaries if s is not None and s != '']
    # rouge1 - unigram based scoring, # rouge 2 - bigram based scoring, rougeL - longest common subsequence based scoring
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    # for each reference summary and each generated summary
    for reference in references:
        for summary in valid_summaries:
            # use the score method to calculate their rouge scores
            scores = scorer.score(reference, summary)
            for metric, score in scores.items():
                rouge_scores[metric].append(score.fmeasure)
    # calculate the average scores for each rouge score across the generated summaries and return the average
    avg_scores = {metric: sum(scores) / len(scores) for metric, scores in rouge_scores.items()}
    return avg_scores

def main():
    """
    main function that loads the finetuned GPT2 model and reads in a test CSV file
    prints the input text, the reference summary, generated summary and ROUGE scores
    """
    # load the fine-tuned model -- change model name if needed
    model = GPT2LMHeadModel.from_pretrained("fine-tuned-model-amaz50kJust-batch64")
    model.eval()
    
    # set up the tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # create empty lists for the input texts and reference summaries
    input_texts = []
    references = []
    # change file to read from if needed -- read in the CSV
    with open("AmazonFineFoods/reviewsTest.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # change indices here to the correct columns for the CSV being used
            input_texts.append(row[1])
            references.append(row[0])
    
    # generate and evaluate summaries for each input text and references
    for input_text, reference in zip(input_texts, references):
        summaries = generate_summaries(input_text, model, tokenizer)
        # check that the list is not empty and if not print the outputs
        if summaries: 
            print("Input Text:", input_text)
            print("Reference Summary:", reference)
            print("Generated Summary:", summaries[0])

            # evaluate the summaries and print the rouge scores
            scores = evaluate_summaries([reference], summaries)
            print()
            print("ROUGE scores:")
            for metric, score in scores.items():
                print(f"{metric}: {score}")
            print()
        # if no valid summaries
        else:
            print("No suitable summary generated.")
        

if __name__ == '__main__':
    main()
