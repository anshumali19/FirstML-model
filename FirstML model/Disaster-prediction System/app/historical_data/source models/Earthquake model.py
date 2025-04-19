import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib  # ✅ To save model
import warnings
import os

# Suppress precision warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Helper function for cleaning coordinates like '20.6N' -> 20.6 or -20.6
def clean_coordinate(coord):
    if isinstance(coord, str):
        direction = coord[-1].upper()
        try:
            value = float(coord[:-1])
            if direction in ['S', 'W']:
                return -value
            return value
        except ValueError:
            return None
    return coord

# Training function
def train_earthquake_model(file, features, label, label_processing=None):
    df = pd.read_csv(file)

    # Clean coordinates
    for col in ['Latitude', 'Longitude']:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].apply(clean_coordinate)

    # Custom label conversion
    if label_processing:
        df[label] = df.apply(label_processing, axis=1)

    # Drop missing values
    df.dropna(subset=features + [label], inplace=True)

    X = df[features]
    y = df[label]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model with class_weight balanced
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(f"\n📋 Classification Report for {label}:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return model

# Get the current directory and paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(current_dir), "data")
app_dir = os.path.dirname(os.path.dirname(current_dir))
models_dir = os.path.join(app_dir, "models_pkl")

# Create models directory if it doesn't exist
os.makedirs(models_dir, exist_ok=True)

# ✅ Train and get the model
earthquake_model = train_earthquake_model(
    file=os.path.join(data_dir, "Earthquake.csv"),
    features=['Magnitude', 'Depth (km)', 'Latitude', 'Longitude'],
    label='quake_risk',
    label_processing=lambda row: 1 if row['Magnitude'] >= 5.0 else 0
)

# Save model in the models_pkl directory
joblib.dump(earthquake_model, os.path.join(models_dir, "earthquake_model.pkl"))