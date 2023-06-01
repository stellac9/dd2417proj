import csv
from tqdm import tqdm

def clean_file(fileFrom, fileTo, summaryIndex, textIndex):
    rows = []

    with open(fileFrom, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in tqdm(csv_reader):
            summary = row[summaryIndex]
            text = row[textIndex]
            
            if len(summary) > 1 and len(text) > 1:
                rows.append([summary, text])

    with open(fileTo, 'w') as csv_file:
         csv_writer = csv.writer(csv_file)
         for row in tqdm(rows):
            csv_writer.writerow(row)


def main():
    clean_file('wiki1.csv','wikiClean.csv', 1, 2)

if __name__ == '__main__':
    main()