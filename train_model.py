import pandas as pd
import os
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# --- 1. Setup and Load Data ---
print("--- Veracity Vigilance: Model Training ---")

# Define paths relative to the project's root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'saved_models', 'minilm_fake_news_model')
MODEL_NAME = 'microsoft/MiniLM-L12-H384-uncased'

# Create directories if they don't exist
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

print("Loading the balanced dataset...")
try:
    real_df = pd.read_csv(os.path.join(DATA_PATH, 'True.csv'))
    fake_df = pd.read_csv(os.path.join(DATA_PATH, 'Fake.csv'))
except FileNotFoundError:
    print(f"❌ Error: Make sure 'True.csv' and 'Fake.csv' are in the '{DATA_PATH}' folder.")
    exit()

# --- 2. Prepare and combine data ---
real_df['label'] = 0  # REAL
fake_df['label'] = 1  # FAKE
df = pd.concat([real_df, fake_df], ignore_index=True)
df['full_text'] = df['title'].astype(str) + ' ' + df['text'].astype(str)
df = df.dropna(subset=['full_text'])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Use a smaller subset for a faster, reliable training session
df_subset = df.head(5000)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_subset['full_text'].tolist(), df_subset['label'].tolist(), test_size=0.2, random_state=42
)

# --- 3. Tokenize the Data ---
print(f"Tokenizing data with '{MODEL_NAME}'...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_encodings, train_labels)
val_dataset = NewsDataset(val_encodings, val_labels)

# --- 4. Train the Model ---
print("Training the MiniLM model on CPU... (This is a one-time process and will take a while)")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Use the compatible argument name for your transformers version
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    logging_steps=100,
    no_cuda=True # Force CPU training
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

# --- 5. Save the Final Model ---
print("✅ Training complete. Saving the final model and tokenizer...")
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
print(f"Model successfully saved to '{MODEL_SAVE_PATH}'")
