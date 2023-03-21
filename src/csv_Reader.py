import csv


def read_in_csv(training_data_path):
    # Open the training data - make sure of correct path
    train_data_file = open(training_data_path)

    # Create a reader to read the csv. Make sure to set the correct delimiter
    # Otherwise the data isnt read correctly
    csvreader = csv.reader(train_data_file, delimiter=';')

    # Extracts the header of the csv
    header = next(csvreader)

    # Extract the rows
    read_in_training_data = []

    # Reads in the training data and converts all entries to ints
    for row in csvreader:
        # Converts all entries  of the actual row to in int
        # and stores it in rows
        read_in_training_data.append([float(x) for x in row])

    return read_in_training_data


def read_in_csv_header(training_data_path):
    # Open the training data - make sure of correct path
    train_data_file = open(training_data_path,encoding='utf-8')

    # Create a reader to read the csv. Make sure to set the correct delimiter
    # Otherwise the data isnt read correctly
    csvreader = csv.reader(train_data_file, delimiter=';')

    # Extracts the header of the csv
    header = next(csvreader)
    return header
