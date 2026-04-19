from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import io
import os
import joblib
from pipeline import predict_pipeline, train_ml_model, apply_rule_based_logic, extract_features
from pipeline import sliding_window_segmentation, preprocess_data, normalize_features

app = Flask(__name__)

# Configuration
MODEL_PATH = 'model/rf_model.pkl'
DATASET_FOLDER = 'dataset'

# ============ HELPER FUNCTIONS ============
def get_model():
    """Load ML model if exists"""
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return None

def validate_csv(df):
    """Validate CSV has required columns"""
    required_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    missing = [col for col in required_cols if col not in df.columns]
    return missing

def truncate_array(arr, max_len=2000):
    """Truncate array for frontend performance"""
    if len(arr) > max_len:
        step = len(arr) // max_len
        return arr[::step][:max_len]
    return arr

# ============ ROUTES ============
@app.route('/')
def index():
    """Render main dashboard"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle CSV upload and run prediction pipeline

    Expected CSV columns: time, lat, lon, speed, ax, ay, az, gx, gy, gz
    Optional: label (for training data)
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Read CSV
        df = pd.read_csv(io.StringIO(file.stream.read().decode('UTF8')))

        # Validate columns
        missing = validate_csv(df)
        if missing:
            return jsonify({
                'error': f'Missing required columns: {", ".join(missing)}. '
                        f'Your CSV has: {", ".join(df.columns.tolist())}'
            }), 400

        # Ensure time column exists
        if 'time' not in df.columns:
            df['time'] = df.index * 20  # Assume 50Hz

        # Ensure speed column exists
        if 'speed' not in df.columns:
            df['speed'] = 10  # Default speed

        # Load ML model
        ml_model = get_model()

        # Run prediction pipeline
        results = predict_pipeline(df, ml_model=ml_model)

        if 'error' in results:
            return jsonify({'error': results['error']}), 400

        # Format for frontend - truncate large arrays
        formatted_results = {
            'timeline': results['timeline'],
            'summary': results['summary'],
            'raw_time': truncate_array(results['raw_time'], 2000),
            'raw_az': truncate_array(results['raw_az'], 2000),
            'raw_gy': truncate_array(results['raw_gy'], 2000),
            'raw_speed': truncate_array(results['raw_speed'], 2000)
        }

        return jsonify(formatted_results)

    except pd.errors.EmptyDataError:
        return jsonify({'error': 'CSV file is empty'}), 400
    except pd.errors.ParserError:
        return jsonify({'error': 'Invalid CSV format. Please check the file.'}), 400
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/train', methods=['POST'])
def train():
    """Train ML model on all CSV files in dataset folder"""
    try:
        if not os.path.exists(DATASET_FOLDER):
            return jsonify({'error': f'Dataset folder "{DATASET_FOLDER}" not found'}), 400

        clf = train_ml_model(DATASET_FOLDER, MODEL_PATH)

        if clf:
            return jsonify({
                'message': 'Model trained successfully',
                'status': 'success',
                'model_path': MODEL_PATH
            })
        else:
            return jsonify({
                'error': 'No valid training data found. Ensure CSV files exist with proper format.',
                'hint': 'Required columns: ax, ay, az, gx, gy, gz, speed (optional: label, time, lat, lon)'
            }), 400

    except Exception as e:
        return jsonify({'error': f'Training failed: {str(e)}'}), 500

@app.route('/predict_single', methods=['POST'])
def predict_single():
    """
    Predict class for a single window of sensor data

    Expected JSON body:
    {
        "ax": [...],
        "ay": [...],
        "az": [...],
        "gx": [...],
        "gy": [...],
        "gz": [...],
        "speed": [...]  (optional)
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Validate
        missing = validate_csv(df)
        if missing:
            return jsonify({'error': f'Missing columns: {", ".join(missing)}'}), 400

        # Add time if not present
        if 'time' not in df.columns:
            df['time'] = df.index * 20

        if 'speed' not in df.columns:
            df['speed'] = 10

        # Extract features
        features = extract_features(df)
        features = normalize_features(features)

        # Apply rule-based logic
        rule_pred = apply_rule_based_logic(features)

        result = {
            'rule_prediction': int(rule_pred),
            'rule_class': ['Smooth', 'Rough', 'Speed Breaker', 'Pothole'][rule_pred] if rule_pred >= 0 else 'Uncertain',
            'features': {
                k: float(v) if isinstance(v, (int, float, np.number)) else v
                for k, v in features.items()
            }
        }

        # Try ML prediction if model exists
        ml_model = get_model()
        if ml_model is not None:
            feat_df = pd.DataFrame([features]).fillna(0)
            expected_cols = ml_model.feature_names_in_ if hasattr(ml_model, 'feature_names_in_') else feat_df.columns
            feat_df = feat_df.reindex(columns=expected_cols, fill_value=0)
            ml_pred = ml_model.predict(feat_df)[0]
            ml_proba = ml_model.predict_proba(feat_df)[0]

            result['ml_prediction'] = int(ml_pred)
            result['ml_class'] = ['Smooth', 'Rough', 'Speed Breaker', 'Pothole'][ml_pred]
            result['ml_confidence'] = float(max(ml_proba))

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_status = 'loaded' if os.path.exists(MODEL_PATH) else 'not_loaded'
    dataset_count = len([f for f in os.listdir(DATASET_FOLDER) if f.endswith('.csv')]) if os.path.exists(DATASET_FOLDER) else 0

    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'dataset_files': dataset_count
    })

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the trained model"""
    if not os.path.exists(MODEL_PATH):
        return jsonify({'error': 'No trained model found'}), 404

    try:
        model = joblib.load(MODEL_PATH)

        info = {
            'model_type': 'RandomForestClassifier',
            'n_estimators': model.n_estimators,
            'max_depth': model.max_depth,
            'n_features': len(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else 'unknown',
            'feature_names': list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else [],
            'classes': [int(c) for c in model.classes_] if hasattr(model, 'classes_') else []
        }

        # Add feature importances if available
        if hasattr(model, 'feature_importances_') and hasattr(model, 'feature_names_in_'):
            importances = dict(zip(model.feature_names_in_, model.feature_importances_))
            info['feature_importances'] = {
                k: float(v) for k, v in sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
            }

        return jsonify(info)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============ ERROR HANDLERS ============
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# ============ MAIN ============
if __name__ == '__main__':
    print("=" * 60)
    print("Road Quality Mapping System")
    print("=" * 60)
    print(f"Model path: {MODEL_PATH}")
    print(f"Dataset folder: {DATASET_FOLDER}")

    # Check model status
    if os.path.exists(MODEL_PATH):
        print("Model status: Trained model found")
    else:
        print("Model status: No trained model (run /train to create one)")

    print("=" * 60)
    print("Starting Flask server on http://127.0.0.1:3000")
    print("=" * 60)

    app.run(debug=True, host='127.0.0.1', port=3000)
