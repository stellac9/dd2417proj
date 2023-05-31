import csv
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from tqdm import tqdm

class Reader():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer 
        self.summaryIndex = 0
        self.textIndex = 2
        self.summaries = []
        self.texts = []

    def read_csv_file(self, file_path):
        summaries = []
        texts = []

        with open(file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # Skip the first row
            #csv_reader.dropna()

            for row in tqdm(csv_reader):
                
                if all(row):
                    summary = " ".join(row[self.summaryIndex].split())
                    text = " ".join(row[self.textIndex].split())
                    #print(summary, text)
                    summaries.append(summary)
                    texts.append(text)

        

        self.summaries = self.tokenizer(summaries, truncation=True, padding=True)["input_ids"]
        self.texts = self.tokenizer(texts, truncation=True, padding=True)["input_ids"]


class CustomDataset(Dataset):
    def __init__(self, texts, summaries):
        self.texts = texts
        self.summaries = summaries

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        #print("index", idx)
        #print("length of texts", len(self.texts))
        #print("length of summaries", len(self.summaries))
       # print("text[idx], summary[idx]", self.texts[idx], self.summaries[idx])
        return self.texts[idx], self.summaries[idx]


def main():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    reader = Reader(tokenizer)
    reader.read_csv_file('wikiHalf.csv')

    # Create GPT-2 model
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Create CustomDataset and DataLoader
    dataset = CustomDataset(reader.texts, reader.summaries)
    dataloader = DataLoader(dataset, batch_size=100)
    #print("length of dataloader", len(dataloader))

    # Training loop
    device = torch.device("cpu")
    model.to(device)
    # 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    num_epochs = 5

    model.train()
    for epoch in range(num_epochs):
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            inputs, labels = batch
            inputs = [input_seq.to(device) for input_seq in inputs]
            labels = [label_seq.to(device) for label_seq in labels]

            # sets gradients of optimised tensors to zero.
            optimizer.zero_grad()
            
            outputs = model(inputs[0], labels=labels[0])
            #print(outputs)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

    # Save the fine-tuned model
    model.save_pretrained("fine-tuned-model")


if __name__ == '__main__':
    main()


# data.dropna() -- removes any rows with NaN values
# head -1000000 fileFrom.csv > fileTo.csv