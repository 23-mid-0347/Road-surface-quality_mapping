import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import folium
from folium.plugins import HeatMap
import warnings
warnings.filterwarnings('ignore')

# ============ CONFIGURATION ============
CLASSES = {
    0: 'Smooth Road',
    1: 'Rough Road',
    2: 'Speed Breaker',
    3: 'Pothole'
}

CLASS_COLORS = {
    0: '#22d3ee',  # Cyan
    1: '#fbbf24',  # Yellow
    2: '#60a5fa',  # Blue
    3: '#f87171'   # Red
}

# ============ DATA PREPROCESSING ============
def preprocess_data(df):
    """Clean and preprocess sensor data"""
    df = df.copy()

    # Remove missing values
    df = df.dropna()

    # Remove stationary data (speed = 0)
    if 'speed' in df.columns:
        df = df[df['speed'] > 0]

    df = df.reset_index(drop=True)

    # Apply smoothing
    cols_to_smooth = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    for col in cols_to_smooth:
        if col in df.columns:
            df[col] = df[col].rolling(window=5, min_periods=1, center=True).mean()

    return df

# ============ FEATURE EXTRACTION ============
def extract_window_features(window):
    """Extract features from a single window of data"""
    features = {}

    # Acceleration features
    for col in ['ax', 'ay', 'az']:
        if col in window.columns:
            data = window[col].values
            features[f'{col}_mean'] = np.mean(data)
            features[f'{col}_std'] = np.std(data)
            features[f'{col}_rms'] = np.sqrt(np.mean(data**2))
            features[f'{col}_p2p'] = np.max(data) - np.min(data)
            features[f'{col}_max'] = np.max(data)
            features[f'{col}_min'] = np.min(data)

            if len(data) > 3:
                features[f'{col}_kurtosis'] = stats.kurtosis(data)
                features[f'{col}_skewness'] = stats.skew(data)
            else:
                features[f'{col}_kurtosis'] = 0
                features[f'{col}_skewness'] = 0

            # Zero crossing rate
            centered = data - np.mean(data)
            zcr = ((centered[:-1] * centered[1:]) < 0).sum()
            features[f'{col}_zcr'] = zcr

    # Gyroscope features
    for col in ['gx', 'gy', 'gz']:
        if col in window.columns:
            data = window[col].values
            features[f'{col}_mean'] = np.mean(data)
            features[f'{col}_std'] = np.std(data)
            features[f'{col}_rms'] = np.sqrt(np.mean(data**2))
            features[f'{col}_p2p'] = np.max(data) - np.min(data)

    # Combined features
    if all(col in window.columns for col in ['ax', 'ay', 'az']):
        acc_mag = np.sqrt(window['ax']**2 + window['ay']**2 + window['az']**2)
        features['acc_mag_mean'] = np.mean(acc_mag)
        features['acc_mag_std'] = np.std(acc_mag)
        features['acc_mag_rms'] = np.sqrt(np.mean(acc_mag**2))

    # Speed features
    if 'speed' in window.columns:
        features['speed_mean'] = window['speed'].mean()
        features['speed_max'] = window['speed'].max()
        features['speed_std'] = window['speed'].std()
    else:
        features['speed_mean'] = 10
        features['speed_max'] = 10
        features['speed_std'] = 0

    return features

# ============ RULE-BASED CLASSIFICATION ============
def rule_based_classifier(features):
    """Classify based on rule-based logic"""
    az_kurtosis = abs(features.get('az_kurtosis', 0))
    az_p2p = features.get('az_p2p', 0)
    az_std = features.get('az_std', 0)
    az_zcr = features.get('az_zcr', 0)

    gy_std = features.get('gy_std', 0)
    gx_std = features.get('gx_std', 0)
    gz_std = features.get('gz_std', 0)
    gyro_total = np.sqrt(gx_std**2 + gy_std**2 + gz_std**2)

    # Pothole: Very high kurtosis + large peak-to-peak
    if az_kurtosis > 8 and az_p2p > 3000:
        return 3

    # Speed Breaker: High gyroscope variation + moderate acceleration
    if gy_std > 150 and az_p2p > 2000:
        return 2

    if gyro_total > 200 and az_p2p > 2500:
        return 2

    # Rough Road: High standard deviation and frequency content
    if az_std > 800 and az_zcr > 3:
        return 1

    # Smooth Road: Low variation
    if az_std < 500 and az_p2p < 1500 and gyro_total < 100:
        return 0

    return -1  # Uncertain

# ============ TRAINING ============
def train_model(data_path='dataset', output_model='model/rf_model.pkl'):
    """Train Random Forest on all available data"""
    print("Loading training data...")

    all_files = []
    for f in os.listdir(data_path):
        if f.endswith('.csv'):
            all_files.append(os.path.join(data_path, f))

    if not all_files:
        print(f"No CSV files found in {data_path}")
        return None

    features_list = []
    labels_list = []

    window_size = 50  # 1 second at 50Hz

    for file_path in all_files:
        print(f"Processing: {os.path.basename(file_path)}")
        try:
            df = pd.read_csv(file_path)
            has_labels = 'label' in df.columns

            df = preprocess_data(df)

            # Create windows
            for i in range(0, len(df) - window_size, window_size // 2):
                window = df.iloc[i:i + window_size]
                if len(window) < window_size // 2:
                    continue

                features = extract_window_features(window)

                if has_labels:
                    label = window['label'].mode().iloc[0]
                else:
                    label = rule_based_classifier(features)

                if label >= 0:
                    features_list.append(features)
                    labels_list.append(label)

        except Exception as e:
            print(f"  Error: {e}")

    if not features_list:
        print("No training data generated!")
        return None

    X = pd.DataFrame(features_list)
    y = np.array(labels_list)

    X = X.fillna(0)

    print(f"\nDataset: {len(X)} samples")
    print(f"Features: {X.shape[1]}")

    unique_labels = np.unique(y)
    print(f"Classes: {dict(zip(*np.unique(y, return_counts=True)))}")

    # If only one class, use rule-based logic to generate diverse labels
    if len(unique_labels) == 1:
        print("Note: Dataset only contains one class. Using rule-based logic for diverse training labels.")
        for i in range(len(features_list)):
            y[i] = rule_based_classifier(features_list[i])
        print(f"Updated classes: {dict(zip(*np.unique(y, return_counts=True)))}")
        unique_labels = np.unique(y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None
    )

    # Train
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced'
    )

    rf.fit(X_train, y_train)

    # Evaluate
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    # Only report on classes that exist
    existing_classes = sorted(unique_labels)
    existing_names = [CLASSES[i] for i in existing_classes]
    print(classification_report(y_test, y_pred, labels=existing_classes, target_names=existing_names))

    # Save
    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    joblib.dump(rf, output_model)
    print(f"\nModel saved to {output_model}")

    return rf

# ============ MAP GENERATION ============
def generate_road_map(csv_file, model_path='model/rf_model.pkl', output_file='road_condition_map.html'):
    """Generate Folium map with road conditions"""
    print(f"Processing {csv_file}...")

    # Load data
    df = pd.read_csv(csv_file)
    df = preprocess_data(df)

    # Load model
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        print("Model not found, using rule-based classification only")
        model = None

    # Process windows
    window_size = 50
    results = []
    coords = []

    for i in range(0, len(df) - window_size, window_size // 2):
        window = df.iloc[i:i + window_size]
        if len(window) < window_size // 2:
            continue

        features = extract_window_features(window)

        # Try rule-based first
        pred = rule_based_classifier(features)

        # Fall back to ML if uncertain
        if pred == -1 and model is not None:
            feat_df = pd.DataFrame([features]).fillna(0)
            expected_cols = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else feat_df.columns
            feat_df = feat_df.reindex(columns=expected_cols, fill_value=0)
            pred = model.predict(feat_df)[0]

        if pred == -1:
            pred = 0  # Default to smooth

        # Get middle coordinate
        mid = window.iloc[len(window)//2]
        if 'lat' in mid and 'lon' in mid:
            coords.append((mid['lat'], mid['lon'], pred))

    if not coords:
        print("No coordinates found in data")
        return

    # Create map
    m = folium.Map(location=[coords[0][0], coords[0][1]], zoom_start=15)

    # Add segments
    for i in range(len(coords) - 1):
        lat1, lon1, pred1 = coords[i]
        lat2, lon2, pred2 = coords[i + 1]

        color = CLASS_COLORS.get(pred1, '#94a3b8')

        folium.PolyLine(
            locations=[(lat1, lon1), (lat2, lon2)],
            color=color,
            weight=5,
            opacity=0.85
        ).add_to(m)

        # Add markers for potholes and speed breakers
        if pred1 in [2, 3]:
            folium.CircleMarker(
                location=(lat1, lon1),
                radius=6,
                color=CLASS_COLORS[pred1],
                fill=True,
                fillOpacity=0.8,
                popup=CLASSES[pred1]
            ).add_to(m)

    # Legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 200px;
                background-color: white; border: 2px solid grey; z-index: 9999;
                font-size: 14px; padding: 15px; border-radius: 8px;">
        <b>Road Condition</b><br>
        <i style="color:#22d3ee;">■</i> Smooth<br>
        <i style="color:#fbbf24;">■</i> Rough<br>
        <i style="color:#60a5fa;">■</i> Speed Breaker<br>
        <i style="color:#f87171;">■</i> Pothole
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(output_file)
    print(f"Map saved to {output_file}")

# ============ MAIN ============
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'map':
        # Generate map from CSV
        csv_file = sys.argv[2] if len(sys.argv) > 2 else 'dataset/bike_data_10_test1.csv'
        generate_road_map(csv_file)
    else:
        # Train model
        train_model()
