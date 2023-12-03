import gzip
import json
import csv
from datetime import datetime
from tqdm import tqdm
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from transformers import BertTokenizer, BertForSequenceClassification
import torch


model_path = './model/bert-base-uncased-sentiment'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)


file_path = './data/steam_reviews.json.gz'

user_dict = set()
item_dict = set()
itemPerUser = defaultdict(list)

# Convert date to int
def convertDate(text):
    match = re.search(r'\b(\w+) (\d+), (\d{4})\b', text)
    if match:
        month_name, day, year = match.groups()
        datetime_obj = datetime.strptime(f"{month_name} {day} {year}", "%B %d %Y")
        return int(datetime_obj.strftime('%Y%m%d'))
    else:
        return None

# # Open the gzip file for reading
# with gzip.open(file_path, 'rt', encoding="utf-8") as file:
#     # Parse the JSON data
#     file.readline()
#     for l in file:
#         # Parse the JSON data
#         data = eval(l)
#
#         # Extract user and steam ID
#         user_id = data['user_id']
#         user_dict.append(user_id)
#
#         # Iterate through each item
#         for item in data['reviews']:
#             # Extract item details
#             item_id = item['item_id']
#             time = convertDate(item['posted'])
#             if time is None:
#                 continue
#             recommend = item['recommend']
#             item_dict.add(item_id)
#             itemPerUser[user_id].append((item_id, time, recommend))
#
#         itemPerUser[user_id] = sorted(itemPerUser[user_id], key=lambda x: x[1])


# Open the gzip file for reading
with gzip.open(file_path, 'rt', encoding="utf-8") as file:
    # Parse the JSON data
    file.readline()
    for l in tqdm(file):
        # Parse the JSON data
        data = eval(l)
        # Extract user and steam ID
        user_id = data['username']
        user_dict.add(user_id)


        # Extract item details
        item_id = data['product_id']
        date_obj = datetime.strptime(data['date'], '%Y-%m-%d')
        time = int(date_obj.strftime('%Y%m%d'))
        text = data['text']
        with torch.no_grad():
            inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            sentiment = 1 if predictions[0, 1] > predictions[0, 0] else 0
            
        item_dict.add(item_id)
        itemPerUser[user_id].append((item_id, time, sentiment))


user_map = {user: index for index, user in enumerate(user_dict)}
item_map = {item: index for index, item in enumerate(item_dict)}

processed_data = []
for u in itemPerUser:
    itemPerUser[u] = sorted(itemPerUser[u], key=lambda x: x[1])
    if len(itemPerUser[u]) >= 6:
        for item in itemPerUser[u]:
            processed_data.append((user_map[u], item_map[item[0]], item[2]))


file_path = './data/data.txt'
with open(file_path, 'w', encoding='utf-8') as file:
    for item in processed_data:
        line = f"{item[0]} {item[1]} {item[2]}\n"
        file.write(line)
