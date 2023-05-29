import csv
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from tqdm import tqdm

class SummarizationDataset(Dataset):
    def __init__(self, texts_encoded, summaries_encoded):
        self.texts_encoded = texts_encoded
        self.summaries_encoded = summaries_encoded

    def __len__(self):
        return len(self.texts_encoded)

    def __getitem__(self, idx):
        return {
            'input_ids': self.texts_encoded[idx],
            'labels': self.summaries_encoded[idx]
        }

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Pad or truncate sequences to a fixed length
    max_length = max(len(seq) for seq in input_ids + labels)
    input_ids = [seq + [0] * (max_length - len(seq)) for seq in input_ids]
    labels = [seq + [-100] * (max_length - len(seq)) for seq in labels]

    return {
        'input_ids': torch.tensor(input_ids),
        'labels': torch.tensor(labels)
    }

class Reader():

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer 
        self.summaryIndex = 0
        self.textIndex = 2
        self.summariesEncoded = []
        self.textsEncoded = []
    
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
                        valueWithoutDoubleSpaces = " ".join(value.split())
                        preparedValue, _ = self.tokenizer.prepare_for_tokenization(valueWithoutDoubleSpaces)
                        self.summariesEncoded.append(self.tokenizer(preparedValue,truncation=True)["input_ids"])
                    elif i ==  self.textIndex:
                        valueWithoutDoubleSpaces = " ".join(value.split())
                        preparedValue, _ = self.tokenizer.prepare_for_tokenization(valueWithoutDoubleSpaces)
                        self.textsEncoded.append(self.tokenizer(preparedValue,truncation=True)["input_ids"])

def main():
    t = GPT2Tokenizer.from_pretrained("gpt2")
    r = Reader(t)
    # read in the wikihow file
    r.read_csv_file('wikihowSep.csv')

    # Prepare dataset
    dataset = SummarizationDataset(r.textsEncoded, r.summariesEncoded)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # Load pre-trained GPT-2 model
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Fine-tuning configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    num_epochs = 3

    # Fine-tuning loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss}")

    # Save the fine-tuned model
    model.save_pretrained("fine_tuned_model")

if __name__ == '__main__':
    main()
