import pandas as pd
import numpy as np
import re
import os

# ----------------------------
# Cleaning Functions
# ----------------------------

def clean_kms(kms):
    try:
        return int(re.sub(r'[^\d]', '', str(kms)))
    except:
        return np.nan

def clean_engine(engine):
    try:
        return int(re.sub(r'[^\d]', '', str(engine)))
    except:
        return np.nan

def extract_power(x):
    try:
        match = re.search(r'([\d.]+)\s*bhp\s*@\s*(\d+)', str(x).lower().replace(',', ''))
        if match:
            return float(match.group(1)), int(match.group(2))
    except:
        return np.nan, np.nan
    return np.nan, np.nan

def extract_torque(x):
    try:
        match = re.search(r'([\d.]+)\s*nm\s*@\s*(\d+)', str(x).lower().replace(',', '').replace('rpm', ''))
        if match:
            return float(match.group(1)), int(match.group(2))
    except:
        return np.nan, np.nan
    return np.nan, np.nan

# ----------------------------
# Main Preprocessing Function
# ----------------------------

def preprocess_car_data(input_path, output_path):
    # Load the dataset
    df = pd.read_csv(input_path)

    # Rename necessary columns
    df.rename(columns={
        'Kilometer': 'Kms_Driven',
        'Engine': 'Engine_Capacity',
        'Max Power': 'Power',
        'Max Torque': 'Torque'
    }, inplace=True)

    # Clean Kms_Driven and Engine
    df['Kms_Driven'] = df['Kms_Driven'].apply(clean_kms)
    df['Engine_Capacity'] = df['Engine_Capacity'].apply(clean_engine)

    # Extract Power and Torque
    df[['max_bhp', 'max_bhp_rpm']] = df['Power'].apply(lambda x: pd.Series(extract_power(x)))
    df[['max_torque', 'max_torque_rpm']] = df['Torque'].apply(lambda x: pd.Series(extract_torque(x)))

    # Convert to appropriate numeric types
    df['max_bhp'] = df['max_bhp'].astype(float)
    df['max_bhp_rpm'] = df['max_bhp_rpm'].astype('Int64')
    df['max_torque'] = df['max_torque'].astype(float)
    df['max_torque_rpm'] = df['max_torque_rpm'].astype('Int64')

    # Save the cleaned dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Cleaned data saved to: {output_path}")


# ----------------------------
# Run Manually (if needed)
# ----------------------------
if __name__ == "__main__":
    preprocess_car_data(
        input_path="data/car details v4.csv",
        output_path="cleaned/cleanedcardata.csv"
    )
