import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning)

# Clean lat/lon if strings
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

def train_cyclone_model(file, features, label):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()  # Strip whitespace

    print("📂 Columns in CSV:", df.columns.tolist())

    for col in ['Latitude', 'Longitude']:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].apply(clean_coordinate)

    df[label] = df.apply(
        lambda row: 1 if row['Wind Speed (km/h)'] >= 100 or row['Pressure (mb)'] <= 980 else 0,
        axis=1
    )

    df.dropna(subset=features + [label], inplace=True)

    X = df[features]
    y = df[label]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

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

# ✅ Train cyclone model and save
cyclone_model = train_cyclone_model(
    file=os.path.join(data_dir, "Cyclone.csv"),
    features=['Wind Speed (km/h)', 'Pressure (mb)', 'Latitude', 'Longitude'],
    label='cyclone_risk'
)

# Save model in the models_pkl directory
joblib.dump(cyclone_model, os.path.join(models_dir, "cyclone_model.pkl"))
