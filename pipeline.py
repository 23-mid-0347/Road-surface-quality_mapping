import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ============ STEP 2: DATA PREPROCESSING ============
def preprocess_data(df):
    """
    Data Preprocessing Steps:
    - Remove rows where speed = 0 (vehicle stationary - introduces noise)
    - Remove missing or NaN values
    - Reset index after cleaning
    - Apply smoothing filter (rolling mean) to reduce sensor noise
    """
    df = df.copy()

    # Remove missing values
    df = df.dropna()

    # Remove speed = 0 (stationary vehicle)
    if 'speed' in df.columns:
        df = df[df['speed'] > 0]

    # Reset index after cleaning
    df = df.reset_index(drop=True)

    # Apply smoothing filter (rolling mean) to reduce sensor noise
    cols_to_smooth = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    for col in cols_to_smooth:
        if col in df.columns:
            df[col] = df[col].rolling(window=5, min_periods=1, center=True).mean()

    return df

# ============ STEP 4: FEATURE ENGINEERING ============
def extract_features(window_df):
    """
    Extract comprehensive features from each window:

    Time-domain features:
    - Mean, Standard Deviation, RMS
    - Peak-to-Peak (max - min)

    Shape features:
    - Kurtosis (detect sharp spikes → potholes)
    - Skewness (detect asymmetry)

    Frequency features:
    - Zero Crossing Rate (detect rough roads)

    Orientation features:
    - Standard deviation of gyroscope pitch (gy)
      → Speed breakers produce clear rotation pattern

    Speed feature:
    - Mean speed of the window
    """
    features = {}

    # Acceleration features (focus on az - vertical acceleration for potholes/bumps)
    for col in ['ax', 'ay', 'az']:
        if col in window_df.columns:
            data = window_df[col].values

            # Time-domain features
            features[f'{col}_mean'] = np.mean(data)
            features[f'{col}_std'] = np.std(data)
            features[f'{col}_rms'] = np.sqrt(np.mean(data**2))
            features[f'{col}_p2p'] = np.max(data) - np.min(data)
            features[f'{col}_max'] = np.max(data)
            features[f'{col}_min'] = np.min(data)

            # Shape features
            if len(data) > 3:
                features[f'{col}_kurtosis'] = stats.kurtosis(data)
                features[f'{col}_skewness'] = stats.skew(data)
            else:
                features[f'{col}_kurtosis'] = 0
                features[f'{col}_skewness'] = 0

            # Frequency features (Zero Crossing Rate)
            centered = data - np.mean(data)
            zcr = ((centered[:-1] * centered[1:]) < 0).sum()
            features[f'{col}_zcr'] = zcr

    # Gyroscope features (rotation detection for speed breakers)
    for col in ['gx', 'gy', 'gz']:
        if col in window_df.columns:
            data = window_df[col].values
            features[f'{col}_mean'] = np.mean(data)
            features[f'{col}_std'] = np.std(data)
            features[f'{col}_rms'] = np.sqrt(np.mean(data**2))
            features[f'{col}_p2p'] = np.max(data) - np.min(data)

    # Combined acceleration magnitude
    if all(col in window_df.columns for col in ['ax', 'ay', 'az']):
        acc_mag = np.sqrt(window_df['ax']**2 + window_df['ay']**2 + window_df['az']**2)
        features['acc_mag_mean'] = np.mean(acc_mag)
        features['acc_mag_std'] = np.std(acc_mag)
        features['acc_mag_rms'] = np.sqrt(np.mean(acc_mag**2))

    # Speed feature
    if 'speed' in window_df.columns:
        features['speed_mean'] = window_df['speed'].mean()
        features['speed_max'] = window_df['speed'].max()
        features['speed_std'] = window_df['speed'].std()
    else:
        features['speed_mean'] = 10  # Default fallback
        features['speed_max'] = 10
        features['speed_std'] = 0

    return features

# ============ STEP 7: SPEED NORMALIZATION ============
def normalize_features(features):
    """
    Normalize vibration features using speed to ensure consistent detection
    across different speeds.

    Formula: normalized_vibration = RMS / (speed + 1)
    """
    speed_mean = features.get('speed_mean', 1)
    if speed_mean <= 0:
        speed_mean = 1

    # Normalize acceleration features by speed
    for col in ['ax', 'ay', 'az']:
        rms_key = f'{col}_rms'
        if rms_key in features:
            features[f'{col}_norm_vibration'] = features[rms_key] / (speed_mean + 1)

    return features

# ============ STEP 5: RULE-BASED CLASSIFICATION (ROUGH SET LOGIC) ============
def apply_rule_based_logic(features, speed_stats=None):
    """
    Rule-Based Classification using Rough Set Logic with Speed Pattern Analysis

    Classes:
    0 = Smooth Road
    1 = Rough Road
    2 = Speed Breaker
    3 = Pothole

    Speed Pattern Analysis:
    - Speed Breaker: Gradual speed decrease before the bump, then recovery
    - Pothole: Sudden jolt with minimal speed change (driver doesn't brake)
    """

    # Extract key features
    az_kurtosis = abs(features.get('az_kurtosis', 0))
    az_p2p = features.get('az_p2p', 0)
    az_std = features.get('az_std', 0)
    az_zcr = features.get('az_zcr', 0)
    az_rms = features.get('az_rms', 0)

    # Gyroscope features for speed breaker detection
    gy_std = features.get('gy_std', 0)
    gy_p2p = features.get('gy_p2p', 0)
    gx_std = features.get('gx_std', 0)
    gz_std = features.get('gz_std', 0)
    gyro_total_std = np.sqrt(gx_std**2 + gy_std**2 + gz_std**2)

    # Speed features
    speed_mean = features.get('speed_mean', 0)
    speed_max = features.get('speed_max', 0)
    speed_std = features.get('speed_std', 0)

    # Speed pattern analysis (if speed_stats provided)
    speed_trend = 0
    if speed_stats:
        speed_start = speed_stats.get('start', speed_mean)
        speed_end = speed_stats.get('end', speed_mean)
        speed_min = speed_stats.get('min', speed_mean)
        speed_trend = speed_end - speed_start  # Negative = deceleration
        speed_drop = speed_max - speed_min if speed_max > speed_min else 0
    else:
        speed_drop = 0
        speed_trend = 0

    # Thresholds calibrated for MPU6050 sensor data
    # Calibrated based on analysis: smooth roads have gy_std ~700-800

    # ========== POTHOLE DETECTION ==========
    # Pothole characteristics:
    # - Very sharp spike (high kurtosis)
    # - Large amplitude change
    # - Sudden jolt with minimal speed change (no braking)
    # - Very short duration

    is_pothole_accel = (az_kurtosis > 3 and az_p2p > 5000) or (az_p2p > 10000 and az_std > 4000)

    # Pothole: Sudden bump WITHOUT significant speed reduction
    # Speed change should be minimal (driver doesn't brake for pothole)
    if is_pothole_accel and speed_drop < 5 and abs(speed_trend) < 3:
        return 3  # Pothole

    if az_p2p > 12000 and az_std > 5000 and speed_drop < 8:
        return 3  # Strong pothole pattern

    # ========== SPEED BREAKER DETECTION ==========
    # Speed breaker characteristics:
    # - Clear rotation pattern in gyroscope (vehicle pitches)
    # - Moderate acceleration spike
    # - Speed gradually decreases before the bump (driver slows down)
    # - Then speed increases after

    # HIGH thresholds - smooth roads have gy_std ~700-800
    is_speed_bumper_gyro = gy_std > 1500 or gyro_total_std > 2000
    is_speed_bumper_accel = az_p2p > 4000 and az_p2p < 12000 and az_std > 2000

    # Speed breaker: Gyro activity WITH speed reduction pattern
    if is_speed_bumper_gyro and is_speed_bumper_accel:
        # Speed pattern: clear deceleration (driver braking before speed bump)
        if speed_drop > 3 or speed_trend < -2:
            return 2  # Speed Breaker
        # Very high gyro activity even without clear speed pattern
        if gy_std > 2500:
            return 2  # Speed Breaker (high confidence from gyro)

    # ========== ROUGH ROAD DETECTION ==========
    # Rough road characteristics:
    # - Sustained high-frequency vibrations
    # - High ZCR (many direction changes)
    # - Moderate but consistent acceleration variation
    # - No clear gyroscope rotation pattern

    is_rough_vibration = az_std > 3500 and az_zcr > 15
    is_rough_sustained = az_p2p > 8000 and az_kurtosis < 2 and gy_std < 1500

    if is_rough_vibration and not is_speed_bumper_gyro:
        return 1  # Rough Road

    if is_rough_sustained and speed_std < 5:
        return 1  # Rough Road (consistent roughness, stable speed)

    # ========== SMOOTH ROAD ==========
    # Smooth road characteristics:
    # - Low acceleration variation
    # - Low gyroscope activity
    # - Stable speed

    if az_std < 3500 and az_p2p < 8000 and gy_std < 1500:
        return 0  # Smooth Road

    # Default: if high gyro but no clear speed pattern, might be rough road
    if gy_std > 1200:
        return 1  # Rough Road (suspected)

    # Otherwise smooth
    return 0

# ============ STEP 3: SLIDING WINDOW SEGMENTATION ============
def sliding_window_segmentation(df, time_col='time', window_ms=1000, overlap=0.5):
    """
    Convert continuous data into windows (events)

    Parameters:
    - window_ms: Window size in milliseconds (1 second = 1000ms)
    - overlap: Overlap between windows (0.5 = 50%)

    Returns list of window dictionaries with start_time, end_time, and data
    """
    windows = []
    if df.empty:
        return windows

    # Ensure time column exists
    if time_col not in df.columns:
        # Create artificial time based on index
        df = df.copy()
        df[time_col] = df.index * 20  # Assume 50Hz = 20ms per sample

    start_time = df[time_col].min()
    end_time = df[time_col].max()

    step_ms = window_ms * (1 - overlap)

    current_start = start_time
    while current_start < end_time:
        current_end = current_start + window_ms
        window_df = df[(df[time_col] >= current_start) & (df[time_col] < current_end)]

        # Need at least 3 samples for meaningful statistics
        if len(window_df) >= 3:
            windows.append({
                'start_time': float(current_start),
                'end_time': float(current_end),
                'data': window_df
            })

        current_start += step_ms

    return windows

# ============ STEP 6: MACHINE LEARNING MODEL ============
def train_ml_model(datasets_folder='dataset', model_path='model/rf_model.pkl'):
    """
    Train Random Forest Classifier on extracted features

    Uses 80/20 train-test split
    Evaluates accuracy and prints classification report
    """
    all_features = []
    all_labels = []

    # Load all csv files in dataset folder
    if not os.path.exists(datasets_folder):
        print(f"Dataset folder '{datasets_folder}' not found.")
        return None

    csv_files = [f for f in os.listdir(datasets_folder) if f.endswith('.csv')]

    if not csv_files:
        print(f"No CSV files found in '{datasets_folder}'")
        return None

    print(f"Loading {len(csv_files)} CSV files...")

    for f in csv_files:
        file_path = os.path.join(datasets_folder, f)
        print(f"Processing: {f}")

        try:
            df = pd.read_csv(file_path)

            # Check for label column
            has_labels = 'label' in df.columns

            # Clean
            df = preprocess_data(df)

            if df.empty:
                print(f"  Warning: No data after preprocessing in {f}")
                continue

            # Segment into windows
            windows = sliding_window_segmentation(df)
            print(f"  Created {len(windows)} windows")

            for w in windows:
                window_df = w['data']
                feat = extract_features(window_df)
                feat = normalize_features(feat)

                if has_labels and 'label' in window_df.columns:
                    # Take the most common label in this window
                    window_label = window_df['label'].mode()
                    if len(window_label) > 0:
                        label = int(window_label.iloc[0])
                        # Ensure label is in valid range [0, 3]
                        if 0 <= label <= 3:
                            all_labels.append(label)
                            all_features.append(feat)
                else:
                    # Use rule-based logic to generate pseudo-labels for training
                    rule_label = apply_rule_based_logic(feat)
                    if rule_label >= 0:
                        all_labels.append(rule_label)
                        all_features.append(feat)

        except Exception as e:
            print(f"  Error processing {f}: {e}")

    if not all_features:
        print("No valid data found to train.")
        return None

    X = pd.DataFrame(all_features)
    y = np.array(all_labels)

    # Fill any remaining NaNs
    X = X.fillna(0)

    print(f"\nTraining dataset: {X.shape[0]} samples, {X.shape[1]} features")

    unique_labels = np.unique(y)
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # If only one class exists in data, use rule-based logic to generate diverse labels
    if len(unique_labels) == 1:
        print("\nNote: Dataset only contains one class. Using rule-based logic for diverse training labels.")
        # Re-generate labels using rule-based logic for all samples
        for i in range(len(all_features)):
            y[i] = apply_rule_based_logic(all_features[i])
        print(f"Updated class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        unique_labels = np.unique(y)

    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )

    # Random Forest Classifier
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'  # Handle imbalanced classes
    )

    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nModel Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    # Only use class names for classes that exist in the data
    existing_classes = sorted(unique_labels)
    class_names = ['Smooth', 'Rough', 'Speed Breaker', 'Pothole']
    existing_names = [class_names[i] for i in existing_classes]
    print(classification_report(y_test, y_pred, labels=existing_classes, target_names=existing_names))

    # Feature importance
    print("\nTop 10 Important Features:")
    importances = pd.Series(clf.feature_importances_, index=X.columns)
    print(importances.nlargest(10).to_string())

    # Save model
    if not os.path.exists('model'):
        os.makedirs('model')
    joblib.dump(clf, model_path)
    print(f"\nModel saved to {model_path}")

    return clf

# ============ STEP 8: PREDICTION PIPELINE ============
def predict_pipeline(df, ml_model=None):
    """
    Complete Prediction Pipeline:
    1. Preprocess data
    2. Segment into windows
    3. Extract features
    4. Apply rule-based logic
    5. If uncertain → fallback to ML model
    6. Output predicted class with confidence

    Returns:
    - timeline: List of window predictions with details
    - summary: Count of each class
    - raw data for visualization
    """

    # Step 1: Preprocess
    df = preprocess_data(df)

    if df.empty:
        return {
            'timeline': [],
            'summary': {0: 0, 1: 0, 2: 0, 3: 0},
            'raw_time': [],
            'raw_az': [],
            'raw_gy': [],
            'raw_speed': [],
            'error': 'No valid data after preprocessing'
        }

    # Step 2: Segment
    windows = sliding_window_segmentation(df)

    # Step 3-5: Feature extraction and classification
    results = []
    events_summary = {0: 0, 1: 0, 2: 0, 3: 0}  # Smooth, Rough, SB, Pothole

    feature_list = []
    predictions = []

    for w in windows:
        window_df = w['data']
        feat = extract_features(window_df)
        feat = normalize_features(feat)
        feature_list.append(feat)

        # Calculate speed statistics for this window
        speed_stats = None
        if 'speed' in window_df.columns and len(window_df) > 1:
            speed_values = window_df['speed'].values
            speed_stats = {
                'start': float(speed_values[0]),
                'end': float(speed_values[-1]),
                'min': float(speed_values.min()),
                'max': float(speed_values.max()),
                'drop': float(speed_values.max() - speed_values.min())
            }

        # Apply rule-based logic with speed pattern analysis
        rule_pred = apply_rule_based_logic(feat, speed_stats)

        # If uncertain and ML model available, use ML
        if rule_pred == -1 and ml_model is not None:
            feat_df = pd.DataFrame([feat]).fillna(0)
            # Ensure column order matches training
            expected_cols = ml_model.feature_names_in_ if hasattr(ml_model, 'feature_names_in_') else feat_df.columns
            feat_df = feat_df.reindex(columns=expected_cols, fill_value=0)
            ml_pred = ml_model.predict(feat_df)[0]
            final_pred = int(ml_pred)
        elif rule_pred == -1:
            # No ML model, default to Smooth
            final_pred = 0
        else:
            final_pred = rule_pred

        predictions.append(final_pred)
        events_summary[final_pred] += 1

        results.append({
            'start_time': float(w['start_time']),
            'end_time': float(w['end_time']),
            'class': final_pred,
            'class_name': ['Smooth', 'Rough', 'Speed Breaker', 'Pothole'][final_pred],
            'az_mean': float(feat.get('az_mean', 0)),
            'az_std': float(feat.get('az_std', 0)),
            'az_p2p': float(feat.get('az_p2p', 0)),
            'az_kurtosis': float(feat.get('az_kurtosis', 0)),
            'gy_std': float(feat.get('gy_std', 0)),
            'gy_p2p': float(feat.get('gy_p2p', 0)),
            'speed_mean': float(feat.get('speed_mean', 0)),
            'norm_vibration': float(feat.get('az_norm_vibration', 0))
        })

    return {
        'timeline': results,
        'summary': events_summary,
        'raw_time': df['time'].tolist() if 'time' in df.columns else list(range(len(df))),
        'raw_az': df['az'].tolist() if 'az' in df.columns else [],
        'raw_gy': df['gy'].tolist() if 'gy' in df.columns else [],
        'raw_speed': df['speed'].tolist() if 'speed' in df.columns else [],
        'features': feature_list,
        'predictions': predictions
    }

# ============ UTILITY FUNCTIONS ============
def load_ml_model(model_path='model/rf_model.pkl'):
    """Load trained ML model if exists"""
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def get_class_color(class_id):
    """Get color for visualization based on class"""
    colors = {
        0: '#22d3ee',  # Smooth - Cyan
        1: '#fbbf24',  # Rough - Yellow
        2: '#60a5fa',  # Speed Breaker - Blue
        3: '#f87171'   # Pothole - Red
    }
    return colors.get(class_id, '#94a3b8')

def get_class_name(class_id):
    """Get human-readable class name"""
    names = {
        0: 'Smooth Road',
        1: 'Rough Road',
        2: 'Speed Breaker',
        3: 'Pothole'
    }
    return names.get(class_id, 'Unknown')

# ============ MAIN (for standalone training) ============
if __name__ == "__main__":
    print("=" * 60)
    print("Road Quality Mapping System - ML Model Training")
    print("=" * 60)
    print("\nClasses:")
    print("  0 = Smooth Road (Green)")
    print("  1 = Rough Road (Yellow)")
    print("  2 = Speed Breaker (Blue)")
    print("  3 = Pothole (Red)")
    print("=" * 60)

    model = train_ml_model()

    if model:
        print("\nTraining complete! Model ready for predictions.")
    else:
        print("\nTraining failed. Check dataset folder.")
