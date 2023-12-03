from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Save locally
model_path = './model/bert-base-uncased-sentiment'
tokenizer.save_pretrained(model_path)
model.save_pretrained(model_path)