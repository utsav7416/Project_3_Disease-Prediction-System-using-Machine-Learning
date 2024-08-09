import pandas as pd
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Generating synthetic data
data = {
    'age': np.random.randint(20, 80, size=n_samples),
    'gender': np.random.choice([0, 1], size=n_samples),  # 0 for female, 1 for male
    'BMI': np.round(np.random.normal(25, 5, size=n_samples), 2),  # normal distribution around 25
    'blood_pressure': np.random.randint(90, 180, size=n_samples),
    'cholesterol': np.random.randint(150, 300, size=n_samples),
    'family_history': np.random.choice([0, 1], size=n_samples),  # 0 for no history, 1 for history
    'disease': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])  # 70% no disease, 30% disease
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('synthetic_health_data.csv', index=False)

print("CSV file 'synthetic_health_data.csv' created.")
