import csv
from tqdm import tqdm


def clean_file(fileFrom, fileTo, summaryIndex, textIndex):
    """ 
    Function that takes two csv files, one to read from and one to write to.
    The one it reads from should have summaries in column with index summaryIndex
    and respective texts in column with index textIndex. Puts this into fileTo
    with summaries in column 0 and texts in column 1. 
    """
    rows = [] #rows to print 

    with open(fileFrom, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in tqdm(csv_reader):
            summary = row[summaryIndex] 
            text = row[textIndex]
            
            if len(summary) > 1 and len(text) > 1: #make sure text/summary is not empty or one char
                rows.append([summary, text])
    
    with open(fileTo, 'w') as csv_file:
         csv_writer = csv.writer(csv_file)
         for row in tqdm(rows):
            csv_writer.writerow(row)


def main():
    clean_file('Reviews.csv','amazClean.csv', 8, 9)

if __name__ == '__main__':
    main()