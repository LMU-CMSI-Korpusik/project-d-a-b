"""
Converts a text file of essays to a CSV file with the following format:
    label, text, generated
"""
    

import csv
import os


def text_to_csv(input_file, output_csv):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_csv, 'a', newline='', encoding='utf-8') as outfile:
        csv_writer = csv.writer(outfile)

        for line in infile:
            # Assuming each line in the text file is a separate field in the CSV
            csv_writer.writerow([0, line.strip(), 1])


# Example usage:
current_script_path = os.path.dirname(__file__)
input_path = f'{current_script_path}/../data/llm_essays.txt'
output_path = f'{current_script_path}/../data/train_essays_llm.csv'
text_to_csv(input_path, output_path)
