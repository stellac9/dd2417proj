import codecs
import numpy as np



class Reader():

    summaryIndex = 8
    textIndex = 9


    def __init__(self):
        self.summaries = np.array([])
        self.texts = np.array([])
        self.summaryIndex = 8
        self.textIndex = 9


    def read_model(self, filename):
        """
        Read a model from file
        """
        with codecs.open(filename, 'r', 'utf-8') as file:
            csvlist = file.read().split('\n')
            for row in csvlist:
                listOfRow = row.split(',')
                if len(listOfRow) > 7: #got weird out of range errors below if this wasn't used (?)
                    self.summaries = np.append(self.summaries, listOfRow[self.summaryIndex])
                    self.texts = np.append(self.texts, listOfRow[self.textIndex])

            


def main():
    r = Reader()
    r.read_model("./Test.csv")
    print(r.summaries)
    print(r.texts)


if __name__ == '__main__':
    main()
