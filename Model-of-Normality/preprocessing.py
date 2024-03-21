import csv
import zipfile 


with zipfile.ZipFile("C:/Users/eleni/Data/300_P.zip", 'r') as zip_ref:
        extracted_data = []
        for file_info in zip_ref.infolist():
            with zip_ref.open(file_info) as file:
                extracted_data.append(file.read())


archive = zipfile.ZipFile("C:/Users/eleni/Data/300_P.zip", 'r')
file = archive.read('300_COVAREP.csv')


with open(file, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        print(', '.join(row))
