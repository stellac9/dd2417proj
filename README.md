# DD2417 - Text Summarization Project

## Group members
Stella Carlsson and David O'Leary

## Installing requirements

Run the following commands for the installations required for the wholeProgram.py file:

      pip install torch torchvision
      pip install transformers
      pip install tqdm

The evaluation files additionally require: 

      pip install rouge-score
      
## Training the model

In the wholeProgram.py file:

Make sure that the path to the CSV file being used for training in the line `reader.read_csv_file('YourCSV.csv')` is correct.

The columns for `self.textIndex` and `self.summaryIndex` should also be changed in the `__init__` function in the `Reader` class if required.

Run the file with `python wholeProgram.py`

## Evaluating the results

The results can be evaluated by either of the following two methods:

1) entering a specific sequence of input text and a reference summary in `evaluate.py` and running the command: `python evaluate.py`

2) replacing the text in `with open("yourCSV.csv", "r") as csvfile:` with the correct name of the CSV being used and ensuring the indices of `input_texts.append` and `references.append` are correct in `evaluateCSV.py`.

The ROUGE scores for the generated summaries will then be printed to the terminal along with a generated summary.
