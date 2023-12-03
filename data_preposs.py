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


if torch.cuda.is_available():
    print("CUDA is available. GPU will be used for inference.")
else:
    print("CUDA is not available. Inference will run on CPU.")

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


model = model.cuda()

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
            
        item_dict.add(item_id)
        itemPerUser[user_id].append((item_id, time, text))


user_map = {user: index for index, user in enumerate(user_dict)}
item_map = {item: index for index, item in enumerate(item_dict)}

cnt = 0
processed_data = []
for u in itemPerUser:
    if len(itemPerUser[u]) >= 6:
        itemPerUser[u] = sorted(itemPerUser[u], key=lambda x: x[1])
        words = [item[2] for item in itemPerUser[u]]
        inputs = tokenizer(words, padding=True, truncation=True, max_length=512, return_tensors="pt").to('cuda')
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu()
            sentiment = [1 if i[1] > i[0] else 0 for i in predictions]
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        for item in itemPerUser[u]:
            cnt += 1
            processed_data.append((user_map[u], item_map[item[0]], sentiment[itemPerUser[u].index(item)]))
            if cnt % 10000 == 0:
                print(cnt)
    else:
        cnt += len(itemPerUser[u])
        if cnt % 10000 == 0:
            print(cnt)
    


file_path = './data/data.txt'
with open(file_path, 'w', encoding='utf-8') as file:
    for item in processed_data:
        line = f"{item[0]} {item[1]} {item[2]}\n"
        file.write(line)
