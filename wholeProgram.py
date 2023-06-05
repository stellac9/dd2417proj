import csv
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from tqdm import tqdm

class Reader():
    def __init__(self, tokenizer):
        """
        initialising variables
        """
        self.tokenizer = tokenizer 
        # update the summaryIndex and textIndex with the correct column indices
        self.summaryIndex = 0
        self.textIndex = 1
        self.summaries = []
        self.texts = []

    def read_csv_file(self, file_path):
        """
        function to read in the CSV
        """
        # create empty lists for summaries and texts
        summaries = []
        texts = []
        # open the csv file 
        with open(file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            # skip the first row of the CSV
            next(csv_reader)
            # for each row in the CSV
            for row in tqdm(csv_reader):
                # if the row is not empty
                if all(row):
                    # get the summary and the text and append them to the lists
                    summary = " ".join(row[self.summaryIndex].split())
                    text = " ".join(row[self.textIndex].split())
                    #print(summary, text)
                    summaries.append(summary)
                    texts.append(text)     
        # pass the tokenised summaries and texts to the self.summaries and self.texts lists
        self.summaries = self.tokenizer(summaries, truncation=True, padding=True)["input_ids"]
        self.texts = self.tokenizer(texts, truncation=True, padding=True)["input_ids"]


class CustomDataset(Dataset):
    def __init__(self, texts, summaries):
        """
        initialise the texts and summaries
        """
        self.texts = texts
        self.summaries = summaries

    def __len__(self):
        """
        return the length of the texts 
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """ 
        returns a tuple with the text and summary at a specific index
        """
        return self.texts[idx], self.summaries[idx]


def main():
    """
    main function for reading the CSV, training and finetuning the model
    """
    # use the pre-trained GPT2 model for the tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # add the PAD token 
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # pass the tokenizer to the reader class and read in the CSV file to be trained on
    reader = Reader(tokenizer)
    reader.read_csv_file('AmazonFineFoods/reviewsMillion.csv')

    # create the GPT2 model from the pretrained GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Create CustomDataset and DataLoaders - setting an appropraite batchsize and loading the data
    dataset = CustomDataset(reader.texts, reader.summaries)
    dataloader = DataLoader(dataset, batch_size=32)

    # use the cpu (or cuda if needed)
    device = torch.device("cpu")
    model.to(device)
    # use Adam to fintune the optimizer by adjusting the learning rate 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    # set the number of epochs
    num_epochs = 3  
    # train the model
    model.train()
    # for each batch in each epoch
    for epoch in range(num_epochs):
        # for each batch in the dataloader
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            # get the inputs and labels from the batch 
            inputs, labels = batch
            inputs = [input_seq.to(device) for input_seq in inputs]
            labels = [label_seq.to(device) for label_seq in labels]

            # sets gradients of the optimized tensors to zero.
            optimizer.zero_grad()
            # get the outputs from the model 
            try:
                outputs = model(inputs[0], labels=labels[0])
            except Exception as e:
                continue
            # calculate the loss from the outputs
            loss = outputs.loss
            # backpropagate to compute the gradients of loss
            loss.backward()
            # updates the model parameters
            optimizer.step()

    # save the fine-tuned model
    model.save_pretrained("fine-tuned-model")


if __name__ == '__main__':
    main()
