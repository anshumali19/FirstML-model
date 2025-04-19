import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib  # ✅ For saving the model as .pkl
import os

# Function to clean coordinates (if needed)
def clean_coordinate(coord):
    try:
        return float(coord)
    except ValueError:
        return None

def train_flood_model(file, features, label):
    # Load data
    df = pd.read_csv(file)

    # Print columns to check for discrepancies
    print("Columns in the DataFrame:", df.columns)

    # Create the flood_risk column if it doesn't exist
    if label not in df.columns:
        df[label] = df['Flood Occurred']  # Ensure label exists

    # Clean coordinates
    for col in ['Latitude', 'Longitude']:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].apply(clean_coordinate)

    # Feature engineering
    df['Rainfall_Water_Level'] = df['Rainfall (mm)'] * df['Water Level (m)']

    # Drop missing values
    df.dropna(subset=features + [label], inplace=True)

    # Features and labels
    X = df[features]
    y = df[label]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # GridSearchCV for RandomForest
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Evaluation
    y_pred = best_model.predict(X_test)
    print(f"\n📋 Classification Report for {label}:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Cross-validation
    cv_scores = cross_val_score(best_model, X_scaled, y, cv=5)
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f}")

    return best_model, scaler

# Get the current directory and paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(current_dir), "data")
app_dir = os.path.dirname(os.path.dirname(current_dir))
models_dir = os.path.join(app_dir, "models_pkl")

# Create models directory if it doesn't exist
os.makedirs(models_dir, exist_ok=True)

# ✅ Train the flood model
flood_model, scaler = train_flood_model(
    file=os.path.join(data_dir, "Flood.csv"),
    features=['Rainfall (mm)', 'Water Level (m)', 'Humidity (%)', 'Latitude', 'Longitude', 'Rainfall_Water_Level'],
    label='flood_risk'
)

# Save models in the models_pkl directory
joblib.dump(flood_model, os.path.join(models_dir, "flood_model.pkl"))
joblib.dump(scaler, os.path.join(models_dir, "flood_scaler.pkl"))
