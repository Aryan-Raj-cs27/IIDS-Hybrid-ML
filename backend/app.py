"""Flask backend for the Intelligent Intrusion Detection System (IIDS).

This backend loads the trained ML artifacts, prepares a small live traffic
simulation pool from the NSL-KDD dataset, and exposes a hybrid inference route
that uses the Random Forest model first and falls back to CNN-LSTM when the RF
confidence is not strong enough.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.models import load_model


# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
# Keep all project locations absolute so the backend works from any launch
# directory inside VS Code, a terminal, or a production deployment.
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, os.pardir))
MODEL_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "model"))
FRONTEND_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "frontend"))
DATA_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, "data", "KDDTrain+_20Percent.txt"))
DB_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, "iids_forensics.db"))

if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)


# -----------------------------------------------------------------------------
# Flask application
# -----------------------------------------------------------------------------
# The frontend folder serves double duty here: it is used for templates and for
# static assets so index.html can be rendered directly from the project UI.
app = Flask(__name__, template_folder=FRONTEND_DIR, static_folder=FRONTEND_DIR)


# -----------------------------------------------------------------------------
# NSL-KDD schema and label mapping
# -----------------------------------------------------------------------------
NSL_KDD_COLUMNS = [
	"duration",
	"protocol_type",
	"service",
	"flag",
	"src_bytes",
	"dst_bytes",
	"land",
	"wrong_fragment",
	"urgent",
	"hot",
	"num_failed_logins",
	"logged_in",
	"num_compromised",
	"root_shell",
	"su_attempted",
	"num_root",
	"num_file_creations",
	"num_shells",
	"num_access_files",
	"num_outbound_cmds",
	"is_host_login",
	"is_guest_login",
	"count",
	"srv_count",
	"serror_rate",
	"srv_serror_rate",
	"rerror_rate",
	"srv_rerror_rate",
	"same_srv_rate",
	"diff_srv_rate",
	"srv_diff_host_rate",
	"dst_host_count",
	"dst_host_srv_count",
	"dst_host_same_srv_rate",
	"dst_host_diff_srv_rate",
	"dst_host_same_src_port_rate",
	"dst_host_srv_diff_host_rate",
	"dst_host_serror_rate",
	"dst_host_srv_serror_rate",
	"dst_host_rerror_rate",
	"dst_host_srv_rerror_rate",
	"label",
	"difficulty",
]


# Map the fine-grained NSL-KDD attack names into the 5 high-level classes used
# by the project. This is the same semantic mapping used during training.
ATTACK_MAP = {
	"normal": "Normal",
	"neptune": "DoS",
	"back": "DoS",
	"land": "DoS",
	"pod": "DoS",
	"smurf": "DoS",
	"teardrop": "DoS",
	"mailbomb": "DoS",
	"apache2": "DoS",
	"processtable": "DoS",
	"udpstorm": "DoS",
	"worm": "DoS",
	"ipsweep": "Probe",
	"nmap": "Probe",
	"portsweep": "Probe",
	"satan": "Probe",
	"mscan": "Probe",
	"saint": "Probe",
	"guess_passwd": "R2L",
	"ftp_write": "R2L",
	"imap": "R2L",
	"phf": "R2L",
	"multihop": "R2L",
	"warezmaster": "R2L",
	"warezclient": "R2L",
	"spy": "R2L",
	"xlock": "R2L",
	"xsnoop": "R2L",
	"snmpguess": "R2L",
	"snmpgetattack": "R2L",
	"httptunnel": "R2L",
	"sendmail": "R2L",
	"named": "R2L",
	"buffer_overflow": "U2R",
	"loadmodule": "U2R",
	"rootkit": "U2R",
	"perl": "U2R",
	"sqlattack": "U2R",
	"xterm": "U2R",
	"ps": "U2R",
}


# LabelEncoder sorts labels alphabetically, so this is the encoded index order
# produced by the training pipeline.
CLASS_NAMES = ["DoS", "Normal", "Probe", "R2L", "U2R"]


# -----------------------------------------------------------------------------
# Global runtime artifacts
# -----------------------------------------------------------------------------
scaler = None
rf_model = None
cnn_lstm_model = None
feature_columns = None
sample_scaled_features = None
sample_protocol_types = None
models_loaded = False
sample_data_loaded = False


def load_sample_data() -> tuple[np.ndarray, np.ndarray]:
	"""Load and preprocess the first 1000 NSL-KDD rows for live simulation.

	The preprocessing mirrors the training pipeline:
	- drop the difficulty column
	- map attack labels to the 5 major classes
	- one-hot encode protocol/service/flag
	- align the resulting feature columns to the saved scaler
	- scale the features using the fitted StandardScaler

	The function returns the scaled feature matrix and the raw protocol_type values
	for each row so the live feed can expose the original traffic protocol.
	"""

	if scaler is None:
		raise RuntimeError("Scaler must be loaded before sample data can be prepared.")

	# Load a small slice of data for a lightweight live traffic pool.
	sample_frame = pd.read_csv(
		DATA_PATH,
		header=None,
		names=NSL_KDD_COLUMNS,
		nrows=1000,
	)

	# Remove the difficulty score immediately, matching the training pipeline.
	sample_frame = sample_frame.drop(columns=["difficulty"])

	# Keep the raw protocol values for the API response before we one-hot encode.
	raw_protocol_types = sample_frame["protocol_type"].astype(str).to_numpy()

	# Apply the exact same NSL-KDD attack grouping used during training.
	sample_frame["label"] = (
		sample_frame["label"].astype(str).str.strip().str.lower().map(ATTACK_MAP)
	)

	if sample_frame["label"].isna().any():
		unknown_labels = sorted(
			sample_frame.loc[sample_frame["label"].isna(), "label"].astype(str).unique()
		)
		raise ValueError(f"Unmapped NSL-KDD labels found in sample pool: {unknown_labels}")

	# Separate the inputs from the target label. The live feed only needs features.
	features = sample_frame.drop(columns=["label"])

	# Perform one-hot encoding on the categorical network fields.
	features = pd.get_dummies(
		features,
		columns=["protocol_type", "service", "flag"],
		drop_first=False,
	)

	# Align the sample feature matrix to the exact column layout the scaler saw
	# during training. Missing columns are filled with zero; extra columns are
	# discarded. This keeps inference shape-safe.
	if feature_columns is not None:
		features = features.reindex(columns=feature_columns, fill_value=0)
	else:
		feature_columns_local = list(getattr(scaler, "feature_names_in_", features.columns))
		features = features.reindex(columns=feature_columns_local, fill_value=0)

	# Apply the fitted scaler to the live sample pool.
	scaled_features = scaler.transform(features)
	return scaled_features, raw_protocol_types


def generate_fake_ip() -> str:
	"""Create a simple fake private network address for the simulation feed."""

	return f"192.168.{np.random.randint(0, 256)}.{np.random.randint(1, 255)}"


def decode_class_index(class_index: int) -> str:
	"""Map a model output index back to the human-readable attack class."""

	if class_index < 0 or class_index >= len(CLASS_NAMES):
		raise ValueError(f"Invalid class index returned by model: {class_index}")
	return CLASS_NAMES[class_index]


# -----------------------------------------------------------------------------
# Database initialization and forensic logging
# -----------------------------------------------------------------------------
def init_db():
	"""Initialize the SQLite forensics database on startup.

	Creates the alerts table if it doesn't exist. This table stores persistent
	logs of all predictions made by the system, including timestamps, source IPs,
	protocols, classifications, confidence scores, and which model made the call.
	"""

	try:
		conn = sqlite3.connect(DB_PATH)
		cursor = conn.cursor()

		cursor.execute(
			"""
			CREATE TABLE IF NOT EXISTS alerts (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				timestamp TEXT NOT NULL,
				source_ip TEXT NOT NULL,
				protocol TEXT NOT NULL,
				classification TEXT NOT NULL,
				confidence REAL NOT NULL,
				model_used TEXT NOT NULL
			)
			"""
		)

		conn.commit()
		print(f"Database initialized at {DB_PATH}")
	except Exception as exc:
		print(f"Error initializing database: {exc}")
	finally:
		conn.close()


def log_alert(timestamp: str, source_ip: str, protocol: str, classification: str, confidence: float, model_used: str):
	"""Log a single prediction result to the forensics database."""

	try:
		conn = sqlite3.connect(DB_PATH)
		cursor = conn.cursor()

		cursor.execute(
			"""
			INSERT INTO alerts (timestamp, source_ip, protocol, classification, confidence, model_used)
			VALUES (?, ?, ?, ?, ?, ?)
			""",
			(timestamp, source_ip, protocol, classification, confidence, model_used),
		)

		conn.commit()
	except Exception as exc:
		print(f"Error logging alert to database: {exc}")
	finally:
		conn.close()


# -----------------------------------------------------------------------------
# Model startup
# -----------------------------------------------------------------------------
# Load the trained scaler and models once at startup so each request can reuse
# the same in-memory artifacts.
try:
	scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
	rf_model_path = os.path.join(MODEL_DIR, "rf_model.pkl")
	cnn_lstm_model_path = os.path.join(MODEL_DIR, "cnn_lstm_model.h5")

	scaler = joblib.load(scaler_path)
	rf_model = joblib.load(rf_model_path)
	cnn_lstm_model = load_model(cnn_lstm_model_path)
	feature_columns = list(getattr(scaler, "feature_names_in_", [])) or None

	# Build the live traffic pool after the scaler is available.
	sample_scaled_features, sample_protocol_types = load_sample_data()

	models_loaded = True
	sample_data_loaded = True
	print("Model artifacts and live sample pool loaded successfully.")
except Exception as exc:
	print(f"Error loading backend artifacts: {exc}")

# Initialize the forensics database on startup
try:
	init_db()
except Exception as exc:
	print(f"Error initializing forensics database: {exc}")


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/")
def index() -> str:
	"""Render the frontend entry page."""

	return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def api_status():
	"""Report backend health and model availability."""

	return jsonify(
		{
			"status": "online",
			"models_loaded": models_loaded,
			"sample_data_loaded": sample_data_loaded,
			"message": "IIDS backend is running.",
		}
	)


@app.route("/api/predict", methods=["POST"])
def api_predict():
	"""Placeholder direct prediction endpoint kept for compatibility."""

	_ = request.get_json(silent=True) or {}
	return jsonify({"status": "success", "prediction": "Normal", "confidence": 0.99})


@app.route("/api/upload_scan", methods=["POST"])
def api_upload_scan():
	"""Scan an uploaded CSV file using hybrid RF/CNN-LSTM inference."""

	if not models_loaded:
		return jsonify({"status": "error", "message": "Models are not loaded."}), 503

	if "file" not in request.files:
		return jsonify({"status": "error", "message": "No file part in request."}), 400

	uploaded_file = request.files["file"]
	if uploaded_file.filename == "":
		return jsonify({"status": "error", "message": "No file selected."}), 400

	try:
		df = pd.read_csv(uploaded_file)
		if df.empty:
			return jsonify({"status": "error", "message": "Uploaded CSV is empty."}), 400

		# Reject exported logs or unrelated CSVs that do not resemble raw NSL-KDD
		# traffic feature inputs.
		required_markers = {"duration", "src_bytes"}
		if not required_markers.intersection(set(df.columns)):
			return (
				jsonify(
					{
						"status": "error",
						"message": "Invalid format. Expected raw network traffic data (41 features), not system logs.",
					}
				),
				400,
			)

		# Accept either feature-only CSVs or raw NSL-KDD rows that include labels.
		if "difficulty" in df.columns:
			df = df.drop(columns=["difficulty"])
		if "label" in df.columns:
			df = df.drop(columns=["label"])

		# Align categorical columns with the training representation.
		if any(column in df.columns for column in ["protocol_type", "service", "flag"]):
			df = pd.get_dummies(
				df,
				columns=[col for col in ["protocol_type", "service", "flag"] if col in df.columns],
				drop_first=False,
			)

		# Match feature shape expected by the trained scaler/model.
		expected_columns = feature_columns or list(getattr(scaler, "feature_names_in_", []))
		if expected_columns:
			df = df.reindex(columns=expected_columns, fill_value=0)

		scaled_data = scaler.transform(df)

		breakdown = {label: 0 for label in CLASS_NAMES}

		# Run hybrid decision logic row by row: RF first, CNN-LSTM fallback.
		for idx, row in enumerate(scaled_data):
			row_2d = row.reshape(1, -1)
			rf_probabilities = rf_model.predict_proba(row_2d)[0]
			rf_confidence = float(np.max(rf_probabilities))

			if rf_confidence >= 0.85:
				predicted_index = int(np.argmax(rf_probabilities))
				model_used = "RF"
				confidence_score = rf_confidence
			else:
				cnn_input = row.reshape(1, 1, row.shape[0])
				cnn_probabilities = cnn_lstm_model.predict(cnn_input, verbose=0)[0]
				predicted_index = int(np.argmax(cnn_probabilities))
				model_used = "CNN-LSTM"
				confidence_score = float(np.max(cnn_probabilities))

			predicted_label = decode_class_index(predicted_index)
			breakdown[predicted_label] += 1

			# Log the prediction to the forensics database
			log_alert(
				timestamp=datetime.now(timezone.utc).isoformat(),
				source_ip=generate_fake_ip(),
				protocol="tcp",  # Generic protocol for uploaded scans
				classification=predicted_label,
				confidence=confidence_score,
				model_used=model_used,
			)

		return jsonify(
			{
				"status": "success",
				"total": int(len(df)),
				"breakdown": breakdown,
			}
		)
	except Exception as exc:
		return (
			jsonify(
				{
					"status": "error",
					"message": f"Invalid or unsupported CSV format: {exc}",
				}
			),
			400,
		)


@app.route("/api/history", methods=["GET"])
def api_history():
	"""Retrieve forensic alert history from the database.

	Returns the last 50 alerts ordered by timestamp (most recent first) as a JSON array.
	This endpoint allows the frontend to display historical detections and perform forensic analysis.
	"""

	try:
		conn = sqlite3.connect(DB_PATH)
		conn.row_factory = sqlite3.Row  # Return rows as dictionaries
		cursor = conn.cursor()

		cursor.execute(
			"""
			SELECT * FROM alerts
			ORDER BY timestamp DESC
			LIMIT 50
			"""
		)

		rows = cursor.fetchall()
		alerts = [dict(row) for row in rows]

		return jsonify(
			{
				"status": "success",
				"count": len(alerts),
				"alerts": alerts,
			}
		)
	except Exception as exc:
		return (
			jsonify(
				{
					"status": "error",
					"message": f"Error retrieving alert history: {exc}",
				}
			),
			500,
		)
	finally:
		conn.close()


@app.route("/api/live_feed", methods=["GET"])
def api_live_feed():
	"""Simulate a live traffic classification event using hybrid inference.

	The route randomly picks one preprocessed NSL-KDD row from the startup pool.
	The Random Forest model makes the first pass. If its confidence is high
	enough, the RF classification is accepted. Otherwise, the same row is passed
	through the CNN-LSTM model for the final decision.
	"""

	if not models_loaded or not sample_data_loaded:
		return (
			jsonify(
				{
					"status": "error",
					"message": "Models or sample data are not loaded.",
				}
			),
			503,
		)

	# Pick a single traffic record from the live simulation pool.
	sample_index = int(np.random.randint(0, sample_scaled_features.shape[0]))
	sample_row = sample_scaled_features[sample_index]
	raw_protocol = str(sample_protocol_types[sample_index])

	# RF expects a 2D array: one sample with the model's feature vector.
	rf_probabilities = rf_model.predict_proba(sample_row.reshape(1, -1))[0]
	rf_class_index = int(np.argmax(rf_probabilities))
	rf_confidence = float(np.max(rf_probabilities))

	# Start with the RF result. If the RF confidence is low, fall back to the
	# CNN-LSTM model, which receives a 3D tensor of shape (1, 1, features).
	model_used = "RF"
	predicted_class_index = rf_class_index
	confidence = rf_confidence

	if rf_confidence < 0.85:
		cnn_input = sample_row.reshape(1, 1, sample_row.shape[0])
		cnn_probabilities = cnn_lstm_model.predict(cnn_input, verbose=0)[0]
		predicted_class_index = int(np.argmax(cnn_probabilities))
		confidence = float(np.max(cnn_probabilities))
		model_used = "CNN-LSTM"

	# Convert the encoded prediction back to the project class label.
	classification = decode_class_index(predicted_class_index)

	# Log this live feed event to the forensics database
	timestamp = datetime.now(timezone.utc).isoformat()
	source_ip = generate_fake_ip()
	log_alert(
		timestamp=timestamp,
		source_ip=source_ip,
		protocol=raw_protocol,
		classification=classification,
		confidence=confidence,
		model_used=model_used,
	)

	return jsonify(
		{
			"timestamp": timestamp,
			"sourceIP": source_ip,
			"protocol": raw_protocol,
			"classification": classification,
			"confidence": confidence,
			"model_used": model_used,
		}
	)


if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5000, debug=True)
