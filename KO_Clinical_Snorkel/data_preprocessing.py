import pandas as pd
import numpy as np
import random
import re
import string
from datetime import timedelta

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Load dataset
file_path = "healthcare_dataset.csv"
df = pd.read_csv(file_path)

# Convert to datetime
df["Date of Admission"] = pd.to_datetime(df["Date of Admission"])
df["Discharge Date"] = pd.to_datetime(df["Discharge Date"])

# =============== STEP 1: Create a realistic medical classification problem ===============

# First, create multiple factors that contribute to the outcome
def generate_medical_factors(row):
    """Generate realistic medical factors that will contribute to an outcome"""
    # Extract basic patient info
    age = row['Age']
    condition = row['Medical Condition'].lower()
    test_result = row['Test Results'].lower()
    stay_duration = (row['Discharge Date'] - row['Date of Admission']).days
    
    # Create risk factors (0-100 scale)
    age_risk = min(100, age * 1.5) if age > 40 else max(0, age * 0.5)
    
    condition_risk = 0
    high_risk_conditions = ['diabetes', 'cancer', 'heart', 'stroke', 'pneumonia', 'copd', 'hypertension']
    medium_risk_conditions = ['arthritis', 'asthma', 'bronchitis', 'depression', 'infection']
    for term in high_risk_conditions:
        if term in condition:
            condition_risk += random.randint(30, 50)
    for term in medium_risk_conditions:
        if term in condition:
            condition_risk += random.randint(10, 25)
    condition_risk = min(100, condition_risk)
    
    # Analyze test results
    result_risk = 0
    if 'normal' in test_result or 'negative' in test_result or 'clear' in test_result:
        result_risk += random.randint(0, 20)
    elif 'abnormal' in test_result or 'positive' in test_result or 'elevated' in test_result:
        result_risk += random.randint(50, 80)
    elif 'inconclusive' in test_result or 'borderline' in test_result:
        result_risk += random.randint(30, 60)
    else:
        # Random baseline risk for unclear results
        result_risk += random.randint(20, 70)
    
    # Stay duration factor
    stay_risk = 0
    if stay_duration < 3:
        stay_risk = random.randint(0, 30)
    elif stay_duration < 7:
        stay_risk = random.randint(20, 50)
    else:
        stay_risk = random.randint(40, 80)
    
    # Add random factors to make this non-deterministic
    random_risk = random.randint(0, 40) - 20  # Range: -20 to +20
    
    # Calculate composite risk score with weights
    total_risk = (
        0.15 * age_risk + 
        0.25 * condition_risk + 
        0.35 * result_risk + 
        0.15 * stay_risk + 
        0.1 * random_risk
    )
    
    # Return all factors
    return {
        'age_factor': age_risk,
        'condition_factor': condition_risk,
        'test_factor': result_risk,
        'stay_factor': stay_risk,
        'random_factor': random_risk,
        'total_risk': total_risk
    }

# Apply function to generate medical factors
risk_factors = df.apply(generate_medical_factors, axis=1)
risk_df = pd.DataFrame(risk_factors.tolist())
df = pd.concat([df, risk_df], axis=1)

# Now determine outcome based on total risk but with significant noise
def determine_outcome(row):
    """Determine outcome based on risk score with reasonable noise"""
    risk = row['total_risk']
    
    # Add substantial noise to create overlapping classes
    noise = random.uniform(-20, 20)
    adjusted_risk = risk + noise
    
    # Determine outcome based on adjusted risk
    if adjusted_risk < 35:
        return "Normal"
    elif adjusted_risk > 65:
        return "Abnormal"
    else:
        return "Inconclusive"

# Apply outcome function
df["Outcome"] = df.apply(determine_outcome, axis=1)

# =============== STEP 2: Create notes with realistic but imperfect correlations ===============

# Define sets of clinical phrases that might appear in notes
clinical_phrases = {
    # Common phrases regardless of outcome
    'common': [
        "Patient was examined", "Medical history reviewed", "Vitals were taken",
        "Patient reports", "Following admission", "Treatment commenced",
        "Labs ordered", "Plan discussed with patient", "Follow-up recommended",
        "Medications prescribed", "Patient tolerated the procedure well"
    ],
    
    # Phrases more common in normal cases, but can appear in any outcome
    'normal_leaning': [
        "within expected parameters", "patient stable", "responding well", 
        "good progress", "favorable response", "reassuring findings",
        "satisfactory recovery", "routine case", "standard presentation"
    ],
    
    # Phrases more common in abnormal cases, but can appear in any outcome
    'abnormal_leaning': [
        "requires monitoring", "deviating from baseline", "worse than previous", 
        "concerning features", "not responding as expected", "unexpected findings",
        "warrants attention", "outside typical values", "suboptimal response"
    ],
    
    # Phrases more common in inconclusive cases, but can appear in any outcome
    'inconclusive_leaning': [
        "difficult to determine", "further evaluation needed", "additional testing required", 
        "mixed findings", "partially resolved", "somewhat improved",
        "unclear if improving", "challenging to interpret", "monitoring recommended"
    ]
}

def create_realistic_note(row):
    """Generate a clinical note with realistic but imperfect correlations to outcomes"""
    # Basic patient info
    age = row['Age']
    gender = row['Gender']
    condition = row['Medical Condition']
    test_results = row['Test Results']
    medication = row['Medication']
    doctor = row['Doctor']
    hospital = row['Hospital']
    admission_date = row['Date of Admission'].strftime("%Y-%m-%d")
    discharge_date = row['Discharge Date'].strftime("%Y-%m-%d")
    outcome = row['Outcome']
    
    # Choose phrases with weighted probability based on outcome
    # This creates correlation but not perfect prediction
    phrases = []
    
    # Always include some common phrases
    phrases.extend(random.sample(clinical_phrases['common'], 2))
    
    # Include outcome-leaning phrases with higher probability for matching outcome
    # But also include some phrases from other outcomes for noise
    if outcome == "Normal":
        # High chance of normal phrases
        if random.random() < 0.8:
            phrases.extend(random.sample(clinical_phrases['normal_leaning'], 2))
        
        # Medium chance of inconclusive phrases
        if random.random() < 0.3:
            phrases.append(random.choice(clinical_phrases['inconclusive_leaning']))
            
        # Low chance of abnormal phrases
        if random.random() < 0.1:
            phrases.append(random.choice(clinical_phrases['abnormal_leaning']))
            
    elif outcome == "Abnormal":
        # High chance of abnormal phrases
        if random.random() < 0.8:
            phrases.extend(random.sample(clinical_phrases['abnormal_leaning'], 2))
        
        # Medium chance of inconclusive phrases
        if random.random() < 0.4:
            phrases.append(random.choice(clinical_phrases['inconclusive_leaning']))
            
        # Low chance of normal phrases
        if random.random() < 0.15:
            phrases.append(random.choice(clinical_phrases['normal_leaning']))
            
    else:  # Inconclusive
        # High chance of inconclusive phrases
        if random.random() < 0.7:
            phrases.extend(random.sample(clinical_phrases['inconclusive_leaning'], 2))
        
        # Medium chance of both normal and abnormal phrases
        if random.random() < 0.5:
            phrases.append(random.choice(clinical_phrases['normal_leaning']))
        
        if random.random() < 0.5:
            phrases.append(random.choice(clinical_phrases['abnormal_leaning']))
    
    # Shuffle phrases to avoid patterns
    random.shuffle(phrases)
    
    # Integrate phrases into templates with varying structures
    templates = [
        # SOAP format
        "S: {age} y/o {gender} presenting with {condition}.\n"
        "O: Admitted on {admission_date} to {hospital}. {phrase1}. Tests showed {test_results}. {phrase2}.\n"
        "A: {phrase3}. Dr. {doctor} prescribed {medication}.\n"
        "P: {phrase4}. Discharged on {discharge_date}.",
        
        # Narrative format
        "Patient is a {age}-year-old {gender} who presented with {condition}. "
        "{phrase1} at {hospital} after admission on {admission_date}. {phrase2}. "
        "Test results: {test_results}. {phrase3}. "
        "Treatment included {medication} as prescribed by Dr. {doctor}. {phrase4}. "
        "Patient was discharged on {discharge_date}.",
        
        # Bullet format
        "- {age}yo {gender}, {condition}\n"
        "- Admitted: {admission_date} to {hospital}\n"
        "- Evaluation: {phrase1}\n"
        "- Tests: {test_results}\n"
        "- {phrase2}\n"
        "- {phrase3}\n"
        "- Treatment: {medication}\n"
        "- {phrase4}\n"
        "- Discharged: {discharge_date}",
        
        # Short format
        "{age}/{gender} with {condition}. {phrase1}. Admitted {admission_date}. "
        "Labs: {test_results}. {phrase2}. {phrase3}. "
        "Tx: {medication}. {phrase4}. DC {discharge_date}."
    ]
    
    # Choose random template
    template_idx = random.randrange(len(templates))
    template = templates[template_idx]
    
    # Create the note
    note = template.format(
        age=age,
        gender=gender,
        condition=condition,
        admission_date=admission_date,
        hospital=hospital,
        test_results=test_results,
        doctor=doctor,
        medication=medication,
        discharge_date=discharge_date,
        phrase1=phrases[0],
        phrase2=phrases[1] if len(phrases) > 1 else "Follow-up scheduled",
        phrase3=phrases[2] if len(phrases) > 2 else "Patient was evaluated",
        phrase4=phrases[3] if len(phrases) > 3 else "Regular follow-up recommended"
    )
    
    # Randomize capitalization and spacing slightly
    if random.random() < 0.3:
        sentences = note.split('. ')
        for i in range(len(sentences)):
            if random.random() < 0.1:
                # Occasionally mess up capitalization
                if len(sentences[i]) > 0:
                    sentences[i] = sentences[i][0].lower() + sentences[i][1:]
        note = '. '.join(sentences)
    
    # Replace some medical terms with abbreviations inconsistently
    medical_abbrevs = {
        'patient': ['pt', 'patient'],
        'history': ['hx', 'history'],
        'treatment': ['tx', 'treatment'],
        'diagnosis': ['dx', 'diagnosis'],
        'symptoms': ['sx', 'symptoms'],
        'discharge': ['dc', 'discharge'],
        'evaluation': ['eval', 'evaluation']
    }
    
    for term, replacements in medical_abbrevs.items():
        if term in note.lower() and random.random() < 0.4:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            repl = random.choice(replacements)
            # Only replace some instances
            count = random.randint(1, 2)
            note = pattern.sub(repl, note, count=count)
    
    # Final check to remove any direct mentions of outcome classes
    note = re.sub(r'\b(normal|abnormal|inconclusive)\b', 'assessed', note, flags=re.IGNORECASE)
    
    return note

# Generate clinical notes
df["Clinical_Note"] = df.apply(create_realistic_note, axis=1)

# =============== STEP 3: Split into old and new datasets and balance classes ===============
df_old = df[(df["Date of Admission"] >= "2019-01-01") & (df["Date of Admission"] <= "2021-12-31")]
df_new = df[(df["Date of Admission"] >= "2022-01-01") & (df["Date of Admission"] <= "2024-12-31")]

def balance_classes(dataframe, column='Outcome', target_per_class=None):
    """Balance classes without exceeding available data"""
    class_counts = dataframe[column].value_counts()
    
    if target_per_class is None:
        samples_per_class = min(class_counts)
    else:
        samples_per_class = min(target_per_class, min(class_counts))
    
    print(f"Sampling {samples_per_class} records per class")
    
    balanced_df = pd.DataFrame()
    for cls in class_counts.index:
        cls_df = dataframe[dataframe[column] == cls]
        sampled = cls_df.sample(min(samples_per_class, len(cls_df)), random_state=42)
        balanced_df = pd.concat([balanced_df, sampled])
    
    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Balance datasets
df_old_balanced = balance_classes(df_old, target_per_class=3000)
df_new_balanced = balance_classes(df_new, target_per_class=1500)

# =============== STEP 4: Validate and analyze dataset ===============

# Print distributions
print("\nOld dataset distribution:")
print(df_old_balanced["Outcome"].value_counts())

print("\nNew dataset distribution:")
print(df_new_balanced["Outcome"].value_counts())

# Check for direct mentions of outcome terms
print("\nChecking for direct mentions of outcome terms:")
for outcome in ["Normal", "Abnormal", "Inconclusive"]:
    old_mentions = sum(df_old_balanced["Clinical_Note"].str.contains(r'\b' + outcome + r'\b', case=False))
    old_percent = old_mentions / len(df_old_balanced) * 100
    
    new_mentions = sum(df_new_balanced["Clinical_Note"].str.contains(r'\b' + outcome + r'\b', case=False))
    new_percent = new_mentions / len(df_new_balanced) * 100
    
    print(f"'{outcome}' appears in {old_mentions} old notes ({old_percent:.2f}%) and {new_mentions} new notes ({new_percent:.2f}%)")

# Check for phrase correlations
print("\nChecking phrase correlations with outcomes:")
for phrase_type in ['normal_leaning', 'abnormal_leaning', 'inconclusive_leaning']:
    # Check first phrase in each category
    check_phrase = clinical_phrases[phrase_type][0]
    
    # Count occurrences by outcome
    for outcome in ["Normal", "Abnormal", "Inconclusive"]:
        outcome_df = df_old_balanced[df_old_balanced["Outcome"] == outcome]
        phrase_count = sum(outcome_df["Clinical_Note"].str.contains(check_phrase, case=False))
        phrase_percent = phrase_count / len(outcome_df) * 100 if len(outcome_df) > 0 else 0
        
        print(f"Phrase '{check_phrase}' appears in {phrase_count} '{outcome}' notes ({phrase_percent:.2f}%)")

# Save datasets
df_old_balanced[["Clinical_Note", "Outcome"]].to_csv("preprocessed_clinical_notes_old.csv", index=False)
df_new_balanced[["Clinical_Note", "Outcome"]].to_csv("preprocessed_clinical_notes_new.csv", index=False)

# Print sample notes
print("\nSample notes from each outcome category:")
for outcome in ["Normal", "Abnormal", "Inconclusive"]:
    sample_row = df_old_balanced[df_old_balanced["Outcome"] == outcome].sample(1).iloc[0]
    
    print(f"\nOutcome: {outcome}")
    print(f"Clinical Note: {sample_row['Clinical_Note']}")
    print(f"Risk Factors: Age={sample_row['age_factor']:.1f}, Condition={sample_row['condition_factor']:.1f}, " +
          f"Test={sample_row['test_factor']:.1f}, Stay={sample_row['stay_factor']:.1f}, Total={sample_row['total_risk']:.1f}")
    print("-" * 80)

print(f"\nOld dataset saved: {len(df_old_balanced)} records")
print(f"New dataset saved: {len(df_new_balanced)} records")