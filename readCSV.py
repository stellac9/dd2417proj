import csv

class Reader():
    # function to read in the CSV file 
    def read_csv_file(self, file_path):
        # store the columns as lists
        column_lists = []
        with open(file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            # getting the number of columns
            num_columns = len(next(csv_reader))
            
            # initialising the column lists as empty
            column_lists = [[] for _ in range(num_columns)]
            
            # reading in, one row at a time, and storing in column_lists 
            for row in csv_reader:
                for i, value in enumerate(row):
                    column_lists[i].append(value)
        
        return column_lists

    # print the first ten lines -- for testing
    def print_first_ten_items(self, column_lists):
        for column in column_lists:
            print(column[:10])


def main():
    r = Reader()
    # read in the wikihow file and store each column in columns
    columns = r.read_csv_file('wikihowSep.csv')
    r.print_first_ten_items(columns)

if __name__ == '__main__':
    main()

