import pandas as pd

# Load the original dataset
df = pd.read_csv("True.csv")

# Keep only 'text' and 'label' columns
df = df[['text', 'label']].dropna()

# Map numeric labels to FAKE and REAL
label_map = {0: 'FAKE', 1: 'REAL'}
df['label'] = df['label'].map(label_map)

# Drop any rows with unmapped labels (just in case)
df = df.dropna(subset=['label'])

# Limit to 2000 rows
df = df.sample(n=2000, random_state=42) if len(df) > 2000 else df

# Save to new CSV
df.to_csv("true_news_dataset.csv", index=False)
print("✅ Saved as news_dataset.csv")
