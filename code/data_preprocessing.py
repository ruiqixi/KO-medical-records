import pandas as pd

# Load dataset
file_path = "healthcare_dataset.csv"  # Replace with actual path
df = pd.read_csv(file_path)

# Convert "Date of Admission" to datetime
df["Date of Admission"] = pd.to_datetime(df["Date of Admission"])

# Function to generate clinical notes
def generate_clinical_note(row):
    note = f"Patient: {row['Name']}, a {row['Age']}-year-old {row['Gender'].lower()} "
    note += f"with a medical history of {row['Medical Condition'].lower()}. "
    note += f"The patient was admitted on {row['Date of Admission'].strftime('%Y-%m-%d')} as a {row['Admission Type'].lower()} case "
    note += f"under the care of {row['Doctor']}. The prescribed medication was {row['Medication'].lower()}. "
    
    if 'Discharge Date' in row and pd.notna(row['Discharge Date']):
        note += f"The patient was discharged on {row['Discharge Date']} "
    
    note += f"with a test result status of {row['Test Results'].lower()}."

    return note

# Apply function to create clinical notes
df["Clinical_Note"] = df.apply(generate_clinical_note, axis=1)

# Rename "Test Results" to "Outcome"
df.rename(columns={"Test Results": "Outcome"}, inplace=True)

# Split dataset based on "Date of Admission"
df_old = df[(df["Date of Admission"] >= "2019-01-01") & (df["Date of Admission"] <= "2021-12-31")]
df_new = df[(df["Date of Admission"] >= "2022-01-01") & (df["Date of Admission"] <= "2024-12-31")]

# Save the split datasets
df_old[["Clinical_Note", "Outcome"]].to_csv("preprocessed_clinical_notes_old.csv", index=False)
df_new[["Clinical_Note", "Outcome"]].to_csv("preprocessed_clinical_notes_new.csv", index=False)

# Display confirmation messages
print(f"Old dataset saved: {len(df_old)} records")
print(f"New dataset saved: {len(df_new)} records")
