import csv
from transformers import GPT2Tokenizer
from tqdm import tqdm

class Reader():

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer 
        self.summaryIndex = 0
        self.textIndex = 3
        self.summaries = []
        self.texts = []
    
    # function to read in the CSV file 
    def read_csv_file(self, file_path):
        with open(file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            # getting rid of first row
            next(csv_reader)

            # reading in, one row at a time, tokenizing the texts and encoding the tokens
            for row in tqdm(csv_reader):
                for i, value in enumerate(row):
                    if i == self.summaryIndex:
                        preparedValue, _ = self.tokenizer.prepare_for_tokenization(value)
                        self.summaries.append(self.tokenizer(preparedValue)["input_ids"])
                    elif i ==  self.textIndex:
                        preparedValue, _ = self.tokenizer.prepare_for_tokenization(value)
                        self.texts.append(self.tokenizer(preparedValue)["input_ids"])



def main():
    t = GPT2Tokenizer.from_pretrained("gpt2")
    r = Reader(t)
    # read in the wikihow file
    r.read_csv_file('wikiSample.csv')

    #print first summary token by token with encoding
    for token_encoding in r.summaries[0]:
        print(token_encoding,t.decode(token_encoding))

if __name__ == '__main__':
    main()

