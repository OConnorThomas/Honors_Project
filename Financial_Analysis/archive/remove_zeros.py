import csv

output_qx_file = 'clean_data/bulk_qx_data.csv'
output_fy_file = 'clean_data/bulk_fy_data.csv'

def contains_zero(row):
    return any(cell.strip() == '0.0' for cell in row)

def filter_csv(input_file, output_file):
    with open(input_file, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        data = [row for row in csvreader if not contains_zero(map(str.strip, row))]

    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(data)

# Filter files
filter_csv(output_qx_file, 'clean_data/filtered_qx_file.csv')
filter_csv(output_fy_file, 'clean_data/filtered_fy_file.csv')