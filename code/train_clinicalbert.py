import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# ========== STEP 1: Load and Preprocess Dataset ==========
print("Loading dataset...")

# Load the old dataset (2019-2021)
file_path = "preprocessed_clinical_notes_old.csv"
df = pd.read_csv(file_path)

# Ensure no missing values
df.dropna(subset=["Clinical_Note", "Outcome"], inplace=True)

# Convert outcomes to numerical labels
label_mapping = {"Normal": 0, "Abnormal": 1, "Inconclusive": 2}
df["label"] = df["Outcome"].map(label_mapping)
# Ensure no data leakage - shuffle before splitting
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
# Split into train and validation sets (80-20 split)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["Clinical_Note"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)

# Load ClinicalBERT tokenizer
tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# Tokenize the text data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

# ========== STEP 2: Convert Data into PyTorch Dataset ==========
class ClinicalNotesDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

# Convert tokenized data into PyTorch dataset
train_dataset = ClinicalNotesDataset(train_encodings, train_labels)
val_dataset = ClinicalNotesDataset(val_encodings, val_labels)

# ========== STEP 3: Load ClinicalBERT Model ==========
print("Loading ClinicalBERT model...")
model = BertForSequenceClassification.from_pretrained(
    "emilyalsentzer/Bio_ClinicalBERT",
    num_labels=3
)

# ========== STEP 4: Set Up Training ==========
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Define loss function and learning rate scheduler
criterion = nn.CrossEntropyLoss()
epochs = 3
num_training_steps = epochs * len(train_loader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# ========== STEP 5: Train the Model ==========
print("Starting training...")
for epoch in range(epochs):
    model.train()
    loop = tqdm(train_loader, leave=True)
    total_loss = 0

    for batch in loop:
        batch = {key: val.to(device) for key, val in batch.items()}

        # Forward pass
        outputs = model(**batch)
        loss = criterion(outputs.logits, batch["labels"])

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # Track loss
        total_loss += loss.item()
        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} completed. Average Loss: {total_loss / len(train_loader)}")

# Save trained model
model.save_pretrained("clinicalbert_patient_outcome")
tokenizer.save_pretrained("clinicalbert_patient_outcome")
print("Model saved successfully.")
# ========== STEP 6: Evaluate on Validation Set ==========
print("Evaluating on validation set...")

model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for batch in val_loader:
        batch = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=-1)
        
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(batch["labels"].cpu().numpy())

# Compute validation accuracy
val_accuracy = accuracy_score(true_labels, predictions)
print(f"Validation Accuracy (Old Dataset 2019-2021): {val_accuracy:.4f}")

# ========== STEP 7: Evaluate on New Dataset ==========
print("Evaluating on new dataset (2022-2024)...")

# Load new dataset
df_test = pd.read_csv("preprocessed_clinical_notes_new.csv")
df_test.dropna(subset=["Clinical_Note", "Outcome"], inplace=True)
df_test["label"] = df_test["Outcome"].map(label_mapping)

test_encodings = tokenizer(df_test["Clinical_Note"].tolist(), truncation=True, padding=True, max_length=512)
test_dataset = ClinicalNotesDataset(test_encodings, df_test["label"].tolist())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

predictions, true_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        batch = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=-1)
        
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(batch["labels"].cpu().numpy())

# Compute test accuracy
test_accuracy = accuracy_score(true_labels, predictions)
print(f"Test Accuracy (New Dataset 2022-2024): {test_accuracy:.4f}")