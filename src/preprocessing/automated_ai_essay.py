"""
Script to generate an essay using the GPT-3.5 API and append it to the training data in csv format.
"""

from openai import OpenAI
import csv
import os
import pandas as pd
import sys

client = OpenAI(
    api_key="YOUR API KEY",
)

current_script_path = os.path.dirname(__file__)
file_path = f'{current_script_path}/../data/train_essays_llm.csv'
prompts_path = f'{current_script_path}/../data/train_prompts.csv'

df = pd.read_csv(prompts_path)

first_row = df.iloc[0]
instructions = first_row['instructions']
source_text = first_row['source_text']
# print(instructions + " " + source_text)

# messages = [{"role": "system", "content":
#              "You are an intelligent assistant."}]

# messages.append(
#     {"role": "user", "content": instructions + " " + source_text}
# )


# chat_completion = client.chat.completions.create(
#     messages=messages,
#     model="gpt-3.5-turbo",
# )

# essay = chat_completion.choices[0].message.content


essay = '\"'.join(sys.argv[1:])
with open(file_path, 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([0, essay, 1])