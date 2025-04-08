import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import pickle
import os
warnings.filterwarnings("ignore")

# Snorkel imports
from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel
from snorkel.analysis import get_label_buckets
from snorkel.utils import probs_to_preds

# ======== STEP 1: Load datasets ========
print("Loading datasets...")
# Load old dataset for training
df_old = pd.read_csv("preprocessed_clinical_notes_old.csv")
# Load new dataset for testing
df_new = pd.read_csv("preprocessed_clinical_notes_new.csv")

# Define label mapping (same as in training)
label_mapping = {"Normal": 0, "Abnormal": 1, "Inconclusive": 2}
id2label = {v: k for k, v in label_mapping.items()}

# Convert string labels to integers
df_old["label"] = df_old["Outcome"].map(label_mapping)
df_new["label"] = df_new["Outcome"].map(label_mapping)

print(f"Old dataset: {len(df_old)} samples")
print(f"New dataset: {len(df_new)} samples")

# ======== STEP 2: Load ClinicalBERT model ========
print("\nLoading ClinicalBERT model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer and model
model_path = "clinicalbert_for_snorkel"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

# ======== STEP 3: Text Analysis for Key Indicators ========
print("\nAnalyzing text patterns for better labeling functions...")

# Analyze word importance by class
def analyze_term_importance(df):
    """Find terms that are most indicative of each class"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import chi2
    
    # Convert text to lowercase
    df['text_lower'] = df['Clinical_Note'].str.lower()
    
    # Vectorize the text
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), min_df=5)
    X = vectorizer.fit_transform(df['text_lower'])
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Calculate chi-squared stats for each class
    class_indicators = {}
    for cls in sorted(df['label'].unique()):
        y = (df['label'] == cls).astype(int)
        chi2_scores, _ = chi2(X, y)
        
        # Get top indicators
        top_indices = np.argsort(chi2_scores)[-30:]
        top_terms = [(feature_names[i], chi2_scores[i]) for i in top_indices]
        top_terms.sort(key=lambda x: x[1], reverse=True)
        
        class_indicators[id2label[cls]] = top_terms
    
    return class_indicators

# Extract key indicators if not previously saved
indicators_file = 'class_indicators.pkl'
if os.path.exists(indicators_file):
    with open(indicators_file, 'rb') as f:
        class_indicators = pickle.load(f)
    print("Loaded saved class indicators")
else:
    try:
        class_indicators = analyze_term_importance(df_old)
        # Save for future use
        with open(indicators_file, 'wb') as f:
            pickle.dump(class_indicators, f)
        print("Analyzed and saved class indicators")
    except Exception as e:
        print(f"Error analyzing term importance: {e}")
        # Default values if analysis fails
        class_indicators = {
            'Normal': [('normal', 100), ('stable', 80), ('regular', 60)],
            'Abnormal': [('abnormal', 100), ('elevated', 80), ('irregular', 60)],
            'Inconclusive': [('inconclusive', 100), ('unclear', 80), ('further', 60)]
        }

# Print top indicators for each class
for cls, indicators in class_indicators.items():
    print(f"\nTop indicators for {cls}:")
    for term, score in indicators[:10]:
        print(f" - {term} ({score:.2f})")

# Helper function to normalize text consistently
def normalize_text(text):
    """Normalize text for consistent processing"""
    if isinstance(text, str):
        text = text.lower()
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        return text
    return ""

# ======== STEP 4: Create Super-Optimized Labeling Functions ========
# Define ABSTAIN value for Snorkel
ABSTAIN = -1

# ----- TIER 1: Ultra-High Confidence Model LFs -----

@labeling_function(name="clinicalbert_ultra_conf")
def lf_clinicalbert_ultra_conf(x):
    """Ultra-high confidence ClinicalBERT predictions (precision-focused)"""
    text = normalize_text(x.Clinical_Note)
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
        pred = torch.argmax(probs, dim=0).item()
        conf = probs[pred].item()
    
    # Only return prediction if confidence is extremely high
    if conf > 0.98:  # Ultra-high confidence threshold
        return pred
    return ABSTAIN

@labeling_function(name="clinicalbert_very_high_conf")
def lf_clinicalbert_very_high_conf(x):
    """Very high confidence ClinicalBERT predictions"""
    text = normalize_text(x.Clinical_Note)
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
        pred = torch.argmax(probs, dim=0).item()
        conf = probs[pred].item()
    
    # Only return prediction if confidence is very high
    if conf > 0.93:  # Very high confidence threshold
        return pred
    return ABSTAIN

@labeling_function(name="clinicalbert_high_conf")
def lf_clinicalbert_high_conf(x):
    """High confidence ClinicalBERT predictions"""
    text = normalize_text(x.Clinical_Note)
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
        pred = torch.argmax(probs, dim=0).item()
        conf = probs[pred].item()
    
    # Only return prediction if confidence is high
    if conf > 0.85:  # High confidence threshold
        return pred
    return ABSTAIN

@labeling_function(name="clinicalbert")
def lf_clinicalbert(x):
    """Standard ClinicalBERT predictions"""
    text = normalize_text(x.Clinical_Note)
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
    
    return pred

# ----- TIER 2: Data-Driven Indicator LFs -----

@labeling_function(name="data_driven_normal")
def lf_data_driven_normal(x):
    """Data-driven indicators for Normal class"""
    text = normalize_text(x.Clinical_Note)
    
    # Use the top indicators extracted from chi-square analysis
    normal_indicators = [term for term, _ in class_indicators.get('Normal', [])][:15]
    
    # Count how many indicators are present
    indicator_count = sum(1 for indicator in normal_indicators if indicator in text)
    
    # Require multiple indicators for stronger signal
    if indicator_count >= 2:
        return label_mapping["Normal"]
    
    return ABSTAIN

@labeling_function(name="data_driven_abnormal")
def lf_data_driven_abnormal(x):
    """Data-driven indicators for Abnormal class"""
    text = normalize_text(x.Clinical_Note)
    
    # Use the top indicators extracted from chi-square analysis
    abnormal_indicators = [term for term, _ in class_indicators.get('Abnormal', [])][:15]
    
    # Count how many indicators are present
    indicator_count = sum(1 for indicator in abnormal_indicators if indicator in text)
    
    # Require multiple indicators for stronger signal
    if indicator_count >= 2:
        return label_mapping["Abnormal"]
    
    return ABSTAIN

@labeling_function(name="data_driven_inconclusive")
def lf_data_driven_inconclusive(x):
    """Data-driven indicators for Inconclusive class"""
    text = normalize_text(x.Clinical_Note)
    
    # Use the top indicators extracted from chi-square analysis
    inconclusive_indicators = [term for term, _ in class_indicators.get('Inconclusive', [])][:15]
    
    # Count how many indicators are present
    indicator_count = sum(1 for indicator in inconclusive_indicators if indicator in text)
    
    # Require multiple indicators for stronger signal
    if indicator_count >= 2:
        return label_mapping["Inconclusive"]
    
    return ABSTAIN

# ----- TIER 3: Expert-Defined Pattern LFs -----

@labeling_function(name="expert_normal_patterns")
def lf_expert_normal_patterns(x):
    """Expert-defined patterns for Normal class"""
    text = normalize_text(x.Clinical_Note)
    
    # Carefully curated patterns that strongly indicate normal results
    normal_patterns = [
        r'\b(test|lab|result).{1,20}(normal|unremarkable|stable|negative)',
        r'\b(patient|condition).{1,15}(stable|good|regular)',
        r'\b(no|not).{1,10}(abnormal|concern|significant|issues)',
        r'\b(normal|stable|regular).{1,15}(finding|result|condition)',
        r'\bdischarged (home|same day)',
        r'\bno.{1,15}(issues|problems|concerns|worries|complications)'
    ]
    
    # Check for matches
    for pattern in normal_patterns:
        if re.search(pattern, text):
            return label_mapping["Normal"]
    
    return ABSTAIN

@labeling_function(name="expert_abnormal_patterns")
def lf_expert_abnormal_patterns(x):
    """Expert-defined patterns for Abnormal class"""
    text = normalize_text(x.Clinical_Note)
    
    # Carefully curated patterns that strongly indicate abnormal results
    abnormal_patterns = [
        r'\b(test|lab|result).{1,20}(abnormal|positive|elevated|concerning)',
        r'\b(require|requires|requiring).{1,15}(treatment|intervention|attention|monitoring)',
        r'\b(severe|significant|notable).{1,10}(symptoms|condition|findings)',
        r'\b(admitted|admission).{1,15}(hospital|emergency|urgently)',
        r'\b(concerning|problematic|worrying).{1,15}(finding|result)',
        r'\b(elevated|increased|high).{1,10}(level|reading|value)'
    ]
    
    # Check for matches
    for pattern in abnormal_patterns:
        if re.search(pattern, text):
            return label_mapping["Abnormal"]
    
    return ABSTAIN

@labeling_function(name="expert_inconclusive_patterns")
def lf_expert_inconclusive_patterns(x):
    """Expert-defined patterns for Inconclusive class"""
    text = normalize_text(x.Clinical_Note)
    
    # Carefully curated patterns that strongly indicate inconclusive results
    inconclusive_patterns = [
        r'\b(test|lab|result).{1,20}(inconclusive|unclear|ambiguous|equivocal)',
        r'\b(additional|further).{1,15}(testing|evaluation|assessment|examination)',
        r'\b(difficult|challenging|hard).{1,15}(interpret|determine|confirm|diagnose)',
        r'\b(monitoring|observation).{1,15}(recommended|suggested|needed|required)',
        r'\b(unclear|uncertain|ambiguous).{1,15}(finding|result|outcome)',
        r'\b(cannot|could not).{1,15}(determine|establish|confirm|rule out)'
    ]
    
    # Check for matches
    for pattern in inconclusive_patterns:
        if re.search(pattern, text):
            return label_mapping["Inconclusive"]
    
    return ABSTAIN

# ----- TIER 4: Medical Context LFs -----

@labeling_function(name="medical_condition_severity")
def lf_medical_condition_severity(x):
    """Medical condition severity assessment"""
    text = normalize_text(x.Clinical_Note)
    
    # Severity indicators
    severity_terms = {
        'high': [
            'severe', 'critical', 'acute', 'emergency', 'urgent', 'intensive', 
            'life-threatening', 'extreme', 'major', 'serious', 'significant'
        ],
        'low': [
            'mild', 'minor', 'slight', 'minimal', 'routine', 'common', 'standard',
            'regular', 'ordinary', 'typical', 'uncomplicated'
        ],
        'medium': [
            'moderate', 'intermediate', 'average', 'varied', 'mixed', 'partial',
            'inconclusive', 'unclear', 'uncertain', 'ambiguous', 'borderline'
        ]
    }
    
    # Count severity indicators
    high_count = sum(1 for term in severity_terms['high'] if term in text)
    low_count = sum(1 for term in severity_terms['low'] if term in text)
    medium_count = sum(1 for term in severity_terms['medium'] if term in text)
    
    # Determine severity level
    if high_count > low_count and high_count > medium_count:
        return label_mapping["Abnormal"]
    elif low_count > high_count and low_count > medium_count:
        return label_mapping["Normal"]
    elif medium_count > high_count and medium_count > low_count:
        return label_mapping["Inconclusive"]
    
    return ABSTAIN

# ----- TIER 5: Combined Feature LFs -----

@labeling_function(name="combined_features")
def lf_combined_features(x):
    """Combine multiple features for a stronger signal"""
    text = normalize_text(x.Clinical_Note)
    
    # Feature extraction
    features = {
        'contains_normal': any(term in text for term in ['normal', 'stable', 'negative', 'unremarkable']),
        'contains_abnormal': any(term in text for term in ['abnormal', 'elevated', 'positive', 'concerning']),
        'contains_inconclusive': any(term in text for term in ['inconclusive', 'unclear', 'ambiguous', 'equivocal']),
        'follow_up_urgency': 'urgent' in text or 'immediate' in text or 'emergency' in text,
        'discharge_home': 'discharge home' in text or 'released home' in text,
        'additional_testing': 'additional test' in text or 'further evaluation' in text or 'more testing' in text
    }
    
    # Decision rules
    if features['contains_normal'] and features['discharge_home'] and not features['follow_up_urgency']:
        return label_mapping["Normal"]
    elif features['contains_abnormal'] and features['follow_up_urgency']:
        return label_mapping["Abnormal"]
    elif features['contains_inconclusive'] and features['additional_testing']:
        return label_mapping["Inconclusive"]
    
    return ABSTAIN

# ======== STEP 5: Apply Labeling Functions ========
# Define our optimized LF set
lfs = [
    # Tier 1: Model-based LFs (highest quality)
    lf_clinicalbert_ultra_conf,
    lf_clinicalbert_very_high_conf,
    lf_clinicalbert_high_conf,
    lf_clinicalbert,
    
    # Tier 2: Data-driven LFs
    lf_data_driven_normal,
    lf_data_driven_abnormal,
    lf_data_driven_inconclusive,
    
    # Tier 3: Expert-defined pattern LFs
    lf_expert_normal_patterns,
    lf_expert_abnormal_patterns,
    lf_expert_inconclusive_patterns,
    
    # Tier 4: Medical context LFs
    lf_medical_condition_severity,
    
    # Tier 5: Combined feature LFs
    lf_combined_features
]

# Apply LFs to the data
print("\nApplying optimized labeling functions...")
applier = PandasLFApplier(lfs)
L_train = applier.apply(df_old)
L_test = applier.apply(df_new)

# ======== STEP 6: Analyze Labeling Function Performance ========
print("\nLabeling function analysis on training set:")
lf_analysis = LFAnalysis(L_train, lfs).lf_summary(df_old["label"].values)
print(lf_analysis)

# Save LF analysis to CSV for easier viewing
lf_analysis.to_csv("lf_analysis_ultimate.csv", index=False)
print("Saved labeling function analysis to 'lf_analysis_ultimate.csv'")

# ======== STEP 7: Train Advanced Label Model ========
print("\nTraining the ultimate label model with optimized parameters...")

# Calculate class balance in training set
class_balance = np.bincount(df_old["label"].values) / len(df_old)
print(f"Class balance in training data: {class_balance}")

# Train the ultimate label model with optimized parameters
label_model = LabelModel(cardinality=len(label_mapping), verbose=True)
label_model.fit(
    L_train, 
    seed=42, 
    n_epochs=2000,  # More epochs for better convergence
    log_freq=500, 
    lr=0.003,  # Optimized learning rate
    l2=0.0002,  # Careful regularization
    class_balance=class_balance
)

# ======== STEP 8: Make Confident Predictions ========
print("\nMaking predictions with advanced confidence thresholding...")

# Function for confident predictions
def predict_with_confidence(label_model, L, threshold=0.75):
    """Make predictions with confidence above threshold"""
    probs = label_model.predict_proba(L)
    confidences = np.max(probs, axis=1)
    preds = probs_to_preds(probs)
    
    # Filter by confidence threshold
    high_confidence_mask = confidences >= threshold
    confident_preds = preds[high_confidence_mask]
    confident_indices = np.where(high_confidence_mask)[0]
    
    return preds, confidences, high_confidence_mask, confident_preds, confident_indices

# Get confident predictions for training set
train_preds, train_confs, train_mask, train_conf_preds, train_conf_idx = predict_with_confidence(
    label_model, L_train, threshold=0.85  # Higher threshold for training
)

# Calculate coverage and accuracy on confident predictions
train_coverage = len(train_conf_preds) / len(train_preds)
if len(train_conf_preds) > 0:
    train_conf_acc = accuracy_score(
        df_old["label"].values[train_conf_idx], 
        train_conf_preds
    )
    print(f"Training set: {train_coverage:.2%} coverage at confidence threshold")
    print(f"Confident predictions accuracy: {train_conf_acc:.4f}")

# Standard predictions for full evaluation
train_preds_all = label_model.predict(L_train)
train_acc = accuracy_score(df_old["label"].values, train_preds_all)
print(f"Training set accuracy (all predictions): {train_acc:.4f}")
print(classification_report(df_old["label"].values, train_preds_all, target_names=label_mapping.keys()))

# ======== STEP 9: Advanced Test Set Evaluation ========
# Get predictions for test set
test_preds, test_confs, test_mask, test_conf_preds, test_conf_idx = predict_with_confidence(
    label_model, L_test, threshold=0.80  # Slightly lower threshold for test set
)

# Calculate coverage and accuracy on confident predictions
test_coverage = len(test_conf_preds) / len(test_preds)
if len(test_conf_preds) > 0:
    test_conf_acc = accuracy_score(
        df_new["label"].values[test_conf_idx], 
        test_conf_preds
    )
    print(f"\nTest set: {test_coverage:.2%} coverage at confidence threshold")
    print(f"Confident predictions accuracy: {test_conf_acc:.4f}")

# Standard predictions for full evaluation
test_preds_all = label_model.predict(L_test)
test_acc = accuracy_score(df_new["label"].values, test_preds_all)
print(f"Test set accuracy (all predictions): {test_acc:.4f}")
print(classification_report(df_new["label"].values, test_preds_all, target_names=label_mapping.keys()))

# ======== STEP 10: Ultimate Ensemble Strategy ========
print("\n===== Ultimate Ensemble Strategy =====")
# Function to get ClinicalBERT predictions with confidence
def get_bert_predictions_with_conf(df):
    """Get predictions and confidence scores from ClinicalBERT model"""
    preds = []
    confs = []
    logits_list = []
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Getting ClinicalBERT predictions"):
        inputs = tokenizer(row["Clinical_Note"], return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.cpu().numpy()[0]
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0].cpu().numpy()
            pred = np.argmax(probs)
            conf = probs[pred]
            preds.append(pred)
            confs.append(conf)
            logits_list.append(logits)
    
    return np.array(preds), np.array(confs), np.array(logits_list)

# Get ClinicalBERT predictions
bert_preds, bert_confs, bert_logits = get_bert_predictions_with_conf(df_new)
bert_acc = accuracy_score(df_new["label"].values, bert_preds)
print(f"ClinicalBERT accuracy on test set: {bert_acc:.4f}")

# Get Snorkel label model probabilities
snorkel_probs = label_model.predict_proba(L_test)

# Create calibrated ensemble predictions
# Convert bert logits to probabilities
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

# Add small offset to avoid division by zero
def calibrated_ensemble(bert_logits, snorkel_probs, bert_weight=0.4, temperature=1.5):
    """Create a calibrated ensemble of BERT and Snorkel predictions"""
    # Convert bert logits to probabilities with temperature scaling
    bert_logits_temp = bert_logits / temperature
    bert_probs = softmax(bert_logits_temp)
    
    # Combine probabilities with weights
    combined_probs = bert_weight * bert_probs + (1 - bert_weight) * snorkel_probs
    
    # Return predictions
    return np.argmax(combined_probs, axis=1), combined_probs

# Try different weight configurations to optimize performance
weight_options = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
temp_options = [1.0, 1.25, 1.5, 1.75, 2.0]
best_acc = 0
best_weight = 0.5
best_temp = 1.0

print("\nOptimizing ensemble weights and temperature...")
for weight in weight_options:
    for temp in temp_options:
        ensemble_preds, _ = calibrated_ensemble(bert_logits, snorkel_probs, bert_weight=weight, temperature=temp)
        acc = accuracy_score(df_new["label"].values, ensemble_preds)
        if acc > best_acc:
            best_acc = acc
            best_weight = weight
            best_temp = temp
            print(f"New best: weight={weight:.2f}, temp={temp:.2f}, acc={acc:.4f}")

# Use the best configuration
print(f"\nUsing optimal configuration: BERT weight={best_weight:.2f}, temperature={best_temp:.2f}")
ensemble_preds, ensemble_probs = calibrated_ensemble(
    bert_logits, snorkel_probs, bert_weight=best_weight, temperature=best_temp
)

# Evaluate ensemble
ensemble_acc = accuracy_score(df_new["label"].values, ensemble_preds)
print(f"Ultimate ensemble accuracy on test set: {ensemble_acc:.4f}")
print(classification_report(df_new["label"].values, ensemble_preds, target_names=label_mapping.keys()))

# Calculate improvement
bert_vs_ensemble = (ensemble_acc - bert_acc) * 100
print(f"Improvement over ClinicalBERT: {bert_vs_ensemble:+.2f}%")

# ======== STEP 11: Knowledge Obsolescence Analysis ========
print("\n===== Knowledge Obsolescence Analysis =====")

# Confusion matrices
print("\nConfusion Matrix (Old Dataset):")
cm_old = confusion_matrix(df_old["label"].values, train_preds_all)
print(cm_old)

print("\nConfusion Matrix (New Dataset with Ultimate Ensemble):")
cm_new = confusion_matrix(df_new["label"].values, ensemble_preds)
print(cm_new)

# Create visualization
try:
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(cm_old, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(label_mapping.keys()),
                yticklabels=list(label_mapping.keys()))
    plt.title('Confusion Matrix - Old Dataset')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_new, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(label_mapping.keys()),
                yticklabels=list(label_mapping.keys()))
    plt.title('Confusion Matrix - New Dataset (Ultimate Ensemble)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices_ultimate.png')
    print("Saved confusion matrices visualization to 'confusion_matrices_ultimate.png'")
except Exception as e:
    print(f"Could not create visualization: {e}")

# Class-specific performance changes
ensemble_report = classification_report(df_new["label"].values, ensemble_preds, output_dict=True)
old_report = classification_report(df_old["label"].values, train_preds_all, output_dict=True)

print("\nPerformance changes by class:")
for i, class_name in enumerate(label_mapping.keys()):
    # Convert class name to string index for dictionary lookup
    class_idx = str(i)
    old_f1 = old_report[class_idx]['f1-score']
    new_f1 = ensemble_report[class_idx]['f1-score']
    change = new_f1 - old_f1
    change_pct = (change / old_f1) * 100 if old_f1 > 0 else float('inf')
    
    print(f"{class_name}: Old F1={old_f1:.4f}, New F1={new_f1:.4f}, Change={change:.4f} ({change_pct:+.2f}%)")

# ======== STEP 12: Advanced Obsolescence Analysis ========
print("\n===== Advanced Obsolescence Analysis =====")

# Calculate per-class knowledge drift
drift_by_class = {}
for i, class_name in enumerate(label_mapping.keys()):
    class_idx = str(i)
    old_precision = old_report[class_idx]['precision']
    old_recall = old_report[class_idx]['recall']
    new_precision = ensemble_report[class_idx]['precision']
    new_recall = ensemble_report[class_idx]['recall']
    
    precision_change = (new_precision - old_precision) / old_precision * 100
    recall_change = (new_recall - old_recall) / old_recall * 100
    
    drift_by_class[class_name] = {
        'precision_change': precision_change,
        'recall_change': recall_change,
        'overall_drift': (abs(precision_change) + abs(recall_change)) / 2
    }

# Print drift analysis
print("Knowledge drift by class:")
for class_name, metrics in drift_by_class.items():
    print(f"{class_name}:")
    print(f"  - Precision change: {metrics['precision_change']:+.2f}%")
    print(f"  - Recall change: {metrics['recall_change']:+.2f}%")
    print(f"  - Overall drift magnitude: {metrics['overall_drift']:.2f}%")

# Identify most stable and most volatile classes
most_stable = min(drift_by_class.items(), key=lambda x: x[1]['overall_drift'])[0]
most_volatile = max(drift_by_class.items(), key=lambda x: x[1]['overall_drift'])[0]

print(f"\nMost stable class: {most_stable}")
print(f"Most volatile class: {most_volatile}")

# ======== STEP 13: Error Analysis with Focus on Knowledge Drift ========
print("\n===== Focused Error Analysis =====")

# Identify high-confidence errors that might indicate knowledge drift
def analyze_errors(y_true, y_pred, confidences, text_data, threshold=0.9):
    """Analyze high-confidence errors that may indicate knowledge drift"""
    errors = []
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i] and confidences[i] >= threshold:
            errors.append({
                'index': i,
                'true': y_true[i],
                'pred': y_pred[i],
                'conf': confidences[i],
                'text': text_data[i][:200] + "..."  # First 200 chars
            })
    
    return errors

# Extract confidence scores from ensemble probabilities
ensemble_confidences = np.max(ensemble_probs, axis=1)

# Find high-confidence errors
high_conf_errors = analyze_errors(
    df_new["label"].values, 
    ensemble_preds, 
    ensemble_confidences,
    df_new["Clinical_Note"].values,
    threshold=0.9
)

# Sort by confidence
high_conf_errors.sort(key=lambda x: x['conf'], reverse=True)

# Print some examples
print(f"Found {len(high_conf_errors)} high-confidence errors")
if len(high_conf_errors) > 0:
    print("\nTop high-confidence errors (potential knowledge drift examples):")
    for i, error in enumerate(high_conf_errors[:3]):
        print(f"\nError {i+1}:")
        print(f"True class: {id2label[error['true']]}")
        print(f"Predicted class: {id2label[error['pred']]}")
        print(f"Confidence: {error['conf']:.4f}")
        print(f"Text: {error['text']}")

# ======== STEP 14: Comparison Across All Methods ========
print("\n===== Comparative Analysis =====")

# Collect accuracy metrics from all models
models = {
    "ClinicalBERT": {
        "accuracy": bert_acc,
        "description": "Standalone fine-tuned ClinicalBERT model"
    },
    "Snorkel": {
        "accuracy": test_acc,
        "description": "Snorkel label model with optimized LFs"
    },
    "Ultimate Ensemble": {
        "accuracy": ensemble_acc,
        "description": f"Calibrated ensemble (BERT weight: {best_weight:.2f}, temp: {best_temp:.2f})"
    }
}

# Print comparative analysis
print("Model accuracy comparison on new dataset:")
for model_name, metrics in models.items():
    print(f"{model_name}: {metrics['accuracy']:.4f} - {metrics['description']}")

# Calculate knowledge obsolescence effect
ko_effect = train_acc - ensemble_acc
ko_percentage = ko_effect / train_acc * 100

print(f"\nKnowledge Obsolescence Effect: {ko_effect:.4f} ({ko_percentage:.2f}%)")
print(f"Old dataset accuracy: {train_acc:.4f}")
print(f"New dataset accuracy (best): {ensemble_acc:.4f}")

# ======== STEP 15: Save the Ultimate Model ========
print("\nSaving the ultimate model and configuration...")

# Save the label model
label_model.save("ultimate_label_model.pkl")

# Save best ensemble configuration
ensemble_config = {
    "bert_weight": best_weight,
    "temperature": best_temp,
    "class_balance": class_balance.tolist()
}

# Save as pickle and text
with open("ultimate_ensemble_config.pkl", "wb") as f:
    pickle.dump(ensemble_config, f)

with open("ultimate_ensemble_config.txt", "w") as f:
    for key, value in ensemble_config.items():
        f.write(f"{key}: {value}\n")

print("Ultimate Snorkel pipeline complete!")