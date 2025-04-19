import joblib
import pandas as pd
import math
import os
from fastapi import FastAPI, Response, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Create templates directory if it doesn't exist
templates_dir = os.path.join(current_dir, "templates")
os.makedirs(templates_dir, exist_ok=True)

# Initialize templates
templates = Jinja2Templates(directory=templates_dir)

# Load your models with absolute paths
flood_model = joblib.load(os.path.join(current_dir, 'models_pkl', 'flood_model.pkl'))
earthquake_model = joblib.load(os.path.join(current_dir, 'models_pkl', 'earthquake_model.pkl'))
cyclone_model = joblib.load(os.path.join(current_dir, 'models_pkl', 'cyclone_model.pkl'))

# Load pincode data
pincode_df = pd.read_csv(os.path.join(current_dir, 'PINCODE(India)', 'pincode_latlon.csv'))
risk_df = pd.read_csv(
    os.path.join(current_dir, 'historical_data', 'data', 'combined_disaster_predictions.csv'),
    dtype={'Latitude': 'float64', 'Longitude': 'float64'},
    low_memory=False
)

# Pydantic models for request body validation
class PredictionRequest(BaseModel):
    latitude: float
    longitude: float

class PincodeRequest(BaseModel):
    pincode: int

# Clean lat/lon if strings (for your data pre-processing)
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

# Distance function
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# Disaster prediction functions
def predict_flood_risk(latitude, longitude):
    features = [latitude, longitude]
    prediction = flood_model.predict([features])
    return {"flood_risk": prediction[0]}

def predict_earthquake_risk(latitude, longitude):
    features = [latitude, longitude]
    prediction = earthquake_model.predict([features])
    return {"earthquake_risk": prediction[0]}

def predict_cyclone_risk(latitude, longitude):
    features = [latitude, longitude]
    prediction = cyclone_model.predict([features])
    return {"cyclone_risk": prediction[0]}

def get_risk_by_pincode(pincode):
    result = pincode_df[pincode_df['pincode'] == pincode]
    
    if result.empty:
        return {"error": "Pincode not found in the dataset"}
    
    lat = float(result.iloc[0]['latitude'])
    lon = float(result.iloc[0]['longitude'])
    location = f"{result.iloc[0].get('officename', 'Unknown')}, {result.iloc[0].get('statename', 'Unknown')}"
    
    tolerance_km = 20  # Increased tolerance to ensure we catch nearby locations
    nearby_risk = []
    
    # Convert risk_df columns to lowercase for consistency
    risk_df.columns = [col.lower() for col in risk_df.columns]
    
    for _, row in risk_df.iterrows():
        risk_lat = row.get('latitude')
        risk_lon = row.get('longitude')
        
        if pd.isna(risk_lat) or pd.isna(risk_lon):
            continue
            
        distance = haversine(lat, lon, risk_lat, risk_lon)
        if distance <= tolerance_km:
            nearby_risk.append(row)
    
    if not nearby_risk:
        return {
            "location": location,
            "coordinates": {"latitude": lat, "longitude": lon},
            "message": "No disaster risk prediction found near this location"
        }
    
    risk_predictions = []
    for row in nearby_risk:
        # Get values with proper column names
        flood_risk = row.get('flood_risk')
        cyclone_risk = row.get('wind speed (km/h)')
        earthquake_risk = row.get('magnitude')
        combined_risk = row.get('combined_risk')
        
        prediction = {
            "location": {"latitude": round(float(row['latitude']), 2), "longitude": round(float(row['longitude']), 2)},
            "flood_risk": round(float(flood_risk), 2) if not pd.isna(flood_risk) else "N/A",
            "cyclone_risk": round(float(cyclone_risk)) if not pd.isna(cyclone_risk) else "N/A",
            "earthquake_risk": round(float(earthquake_risk), 1) if not pd.isna(earthquake_risk) else "N/A",
            "combined_risk": int(round(float(combined_risk))) if not pd.isna(combined_risk) else "N/A"
        }
        risk_predictions.append(prediction)
    
    return {
        "location": location,
        "coordinates": {"latitude": lat, "longitude": lon},
        "predictions": risk_predictions
    }

# Handle favicon.ico requests
@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return Response(status_code=204)

# Predict disaster risk for a specific location
@app.post("/predict_flood_risk")
def predict_flood(data: PredictionRequest):
    return predict_flood_risk(data.latitude, data.longitude)

@app.post("/predict_earthquake_risk")
def predict_earthquake(data: PredictionRequest):
    return predict_earthquake_risk(data.latitude, data.longitude)

@app.post("/predict_cyclone_risk")
def predict_cyclone(data: PredictionRequest):
    return predict_cyclone_risk(data.latitude, data.longitude)

@app.post("/predict_by_pincode")
def predict_by_pincode(data: PincodeRequest):
    return get_risk_by_pincode(data.pincode)

# Create HTML template
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Disaster Risk Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .form-group {
            margin-bottom: 20px;
        }
        input[type="number"] {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            width: 100%;
        }
        button:hover {
            background-color: #45a049;
        }
        #result {
            margin-top: 20px;
            display: none;
        }
        .prediction {
            margin-top: 10px;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .error {
            color: red;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Disaster Risk Prediction</h1>
        <div class="form-group">
            <label for="pincode">Enter Pincode:</label>
            <input type="number" id="pincode" placeholder="Enter 6-digit pincode">
        </div>
        <button onclick="getPrediction()">Get Prediction</button>
        <div id="result"></div>
    </div>

    <script>
        async function getPrediction() {
            const pincode = document.getElementById('pincode').value;
            if (!pincode || pincode.length !== 6) {
                alert('Please enter a valid 6-digit pincode');
                return;
            }

            try {
                const response = await fetch('/predict_by_pincode', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ pincode: parseInt(pincode) })
                });
                const data = await response.json();
                displayResult(data);
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('result').innerHTML = '<div class="error">Error fetching prediction. Please try again.</div>';
            }
        }

        function displayResult(data) {
            const resultDiv = document.getElementById('result');
            resultDiv.style.display = 'block';

            if (data.error) {
                resultDiv.innerHTML = `<div class="error">${data.error}</div>`;
                return;
            }

            let html = `
                <h2>Location: ${data.location}</h2>
                <p>Coordinates: Latitude ${data.coordinates.latitude}, Longitude ${data.coordinates.longitude}</p>
            `;

            if (data.message) {
                html += `<p>${data.message}</p>`;
            } else if (data.predictions && data.predictions.length > 0) {
                html += '<h3>Disaster Risk Predictions:</h3>';
                data.predictions.forEach(pred => {
                    html += `
                        <div class="prediction">
                            <p>Location (approx): (${pred.location.latitude}, ${pred.location.longitude})</p>
                            <p>Flood Risk: ${pred.flood_risk}</p>
                            <p>Cyclone Risk: ${pred.cyclone_risk}</p>
                            <p>Earthquake Risk: ${pred.earthquake_risk}</p>
                            <p>Combined Risk (Model): ${pred.combined_risk}</p>
                        </div>
                    `;
                });
            }

            resultDiv.innerHTML = html;
        }
    </script>
</body>
</html>
"""

# Create the template file
with open(os.path.join(templates_dir, "index.html"), "w") as f:
    f.write(html_content)

# Update the root endpoint to serve the HTML page
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
