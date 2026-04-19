# Road Quality Mapping System

A complete system for detecting and classifying road conditions using bike sensor data (accelerometer, gyroscope, and speed).

## Features

- **4-Class Classification**:
  - 0: Smooth Road (Green)
  - 1: Rough Road (Yellow)  
  - 2: Speed Breaker (Blue)
  - 3: Pothole (Red)

- **Hybrid Classification**: Combines rule-based logic (Rough Set) with Machine Learning (Random Forest)
- **Speed Normalization**: Ensures consistent detection across different vehicle speeds
- **Interactive Dashboard**: Real-time visualization with Chart.js
- **Folium Map**: Geographic visualization of road conditions

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Sensor CSV    │────▶│  Preprocessing   │────▶│ Sliding Window  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                           │
                                                           ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Dashboard UI   │◀────│   Prediction     │◀────│ Feature Extract │
│  (Chart.js)     │     │   Pipeline       │     │ (Statistical)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
           ┌──────────────┐    ┌────────────────┐
           │ Rule-Based   │    │ ML Model       │
           │ (Rough Set)  │    │ (Random Forest)│
           └──────────────┘    └────────────────┘
```

## Installation

```bash
# Install dependencies
pip install flask pandas numpy scipy scikit-learn joblib folium

# Or use requirements.txt
pip install -r requirements.txt
```

## Usage

### 1. Start the Web Dashboard

```bash
python app.py
```

Open browser: http://localhost:5000

### 2. Upload CSV File

Drag and drop or click to select a CSV file with columns:
- Required: `ax, ay, az, gx, gy, gz`
- Recommended: `time, speed, lat, lon`

### 3. Train ML Model (Optional)

Click "Train ML Model" to train on all CSV files in the `dataset/` folder.

### 4. View Results

- Summary cards showing counts of each road condition
- Acceleration (az) and Gyroscope (gy) charts with event markers
- Timeline of road conditions
- Classification distribution pie chart
- Detailed events table

## CSV Format

```csv
time,lat,lon,speed,ax,ay,az,acc_mag,gx,gy,gz,label
237279,12.969813,79.164851,15.5,-2056,-804,15712,15866.33,-30,512,84,0
...
```

### Columns Explained:
- `time`: Timestamp in milliseconds
- `lat`, `lon`: GPS coordinates (for map visualization)
- `speed`: Vehicle speed
- `ax, ay, az`: Accelerometer data (X, Y, Z axes)
- `gx, gy, gz`: Gyroscope data (X, Y, Z axes)
- `acc_mag`: Acceleration magnitude (optional)
- `label`: Ground truth label (optional, 0-3)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard |
| `/upload` | POST | Upload CSV for analysis |
| `/train` | POST | Train ML model |
| `/predict_single` | POST | Predict single window |
| `/health` | GET | Health check |
| `/model_info` | GET | Model information |

## File Structure

```
road_quality/
├── app.py              # Flask web application
├── pipeline.py         # Core prediction pipeline
├── ml.py              # ML training and map generation
├── templates/
│   └── index.html     # Interactive dashboard
├── dataset/           # Training data (CSV files)
├── model/            # Saved ML models
└── outputs/          # Generated maps
```

## Detection Algorithms

### Rule-Based Logic (Rough Set)

1. **Pothole Detection**: High kurtosis (>2) + Large peak-to-peak (>5000)
2. **Speed Breaker**: High gyroscope variation (>100) + Moderate acceleration
3. **Rough Road**: Sustained high standard deviation (>2500)
4. **Smooth Road**: Low variation across all metrics

### Feature Extraction

**Time-Domain Features**:
- Mean, Standard Deviation, RMS
- Peak-to-Peak (max - min)

**Shape Features**:
- Kurtosis (detects sharp spikes)
- Skewness (detects asymmetry)

**Frequency Features**:
- Zero Crossing Rate

**Orientation Features**:
- Gyroscope pitch/roll variation

**Speed Features**:
- Mean speed
- Normalized vibration: `RMS / (speed + 1)`

### Machine Learning

- Algorithm: Random Forest Classifier
- Estimators: 200
- Features: 30+ statistical features
- Train/Test Split: 80/20
- Class Weights: Balanced

## Calibration

The system uses thresholds calibrated for MPU6050 sensor data (typical Arduino bike setup). If using different sensors, adjust thresholds in `pipeline.py`:

```python
# Pothole thresholds
if az_kurtosis > 2 and az_p2p > 5000:
    return 3

# Speed breaker thresholds  
if gy_std > 100 and az_p2p > 4000:
    return 2

# Rough road thresholds
if az_std > 2500 and az_zcr > 10:
    return 1
```

## Development

### Adding New Detection Rules

Edit `pipeline.py` function `apply_rule_based_logic()`:

```python
def apply_rule_based_logic(features):
    # Add your custom rules here
    if your_condition:
        return class_id
    return -1  # Uncertain (use ML)
```

### Training on Custom Data

1. Add labeled CSV files to `dataset/` folder
2. Ensure CSV has a `label` column (0-3)
3. Run training: `python pipeline.py`

## License

MIT License
