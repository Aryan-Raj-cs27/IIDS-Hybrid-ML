"""Training pipeline for the Intelligent Intrusion Detection System.

This module loads NSL-KDD training data, preprocesses it for both a classical
machine learning baseline and a deep learning hybrid model, trains the models,
and persists the fitted artifacts.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv1D, Dense, Dropout, LSTM, MaxPooling1D
from tensorflow.keras.models import Sequential


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "KDDTrain+_20Percent.txt"
SCALER_PATH = BASE_DIR / "scaler.pkl"
RF_MODEL_PATH = BASE_DIR / "rf_model.pkl"
CNN_LSTM_MODEL_PATH = BASE_DIR / "cnn_lstm_model.h5"


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


def load_data() -> pd.DataFrame:
	"""Load the NSL-KDD training dataset with the standard column layout."""

	try:
		data = pd.read_csv(DATA_PATH, header=None, names=NSL_KDD_COLUMNS)
	except FileNotFoundError as exc:
		raise FileNotFoundError(
			f"Dataset not found at {DATA_PATH}. Verify the NSL-KDD file is present."
		) from exc
	except pd.errors.ParserError as exc:
		raise ValueError(f"Failed to parse dataset at {DATA_PATH}: {exc}") from exc

	return data.drop(columns=["difficulty"])


def preprocess_data(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, StandardScaler, LabelEncoder]:
	"""Transform the raw NSL-KDD data into model-ready arrays."""

	processed = data.copy()
	processed["label"] = processed["label"].astype(str).str.strip().str.lower().map(ATTACK_MAP)

	if processed["label"].isna().any():
		unknown_labels = sorted(data.loc[processed["label"].isna(), "label"].astype(str).unique())
		raise ValueError(f"Unmapped NSL-KDD attack labels found: {unknown_labels}")

	features = processed.drop(columns=["label"])
	target = processed["label"]

	target_encoder = LabelEncoder()
	encoded_target = target_encoder.fit_transform(target)

	categorical_columns = ["protocol_type", "service", "flag"]
	features = pd.get_dummies(features, columns=categorical_columns, drop_first=False)

	scaler = StandardScaler()
	scaled_features = scaler.fit_transform(features)
	joblib.dump(scaler, SCALER_PATH)

	return scaled_features, encoded_target, scaler, target_encoder


def train_rf(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> RandomForestClassifier:
	"""Train and evaluate the Random Forest baseline model."""

	rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
	rf_model.fit(x_train, y_train)

	predictions = rf_model.predict(x_test)
	accuracy = accuracy_score(y_test, predictions)
	print(f"Random Forest Accuracy: {accuracy:.4f}")

	joblib.dump(rf_model, RF_MODEL_PATH)
	return rf_model


def reshape_for_dl(features: np.ndarray) -> np.ndarray:
	"""Reshape 2D tabular input into the 3D tensor expected by Conv1D/LSTM."""

	if features.ndim != 2:
		raise ValueError(f"Expected a 2D array for reshaping, received shape {features.shape}.")

	return features.reshape(features.shape[0], 1, features.shape[1])


def build_cnn_lstm_model(input_shape: tuple[int, int]) -> Sequential:
	"""Build the CNN-LSTM architecture for hybrid sequence learning."""

	model = Sequential(
		[
			Conv1D(filters=64, kernel_size=1, activation="relu", input_shape=input_shape),
			MaxPooling1D(pool_size=1),
			LSTM(64),
			Dropout(0.2),
			Dense(5, activation="softmax"),
		]
	)
	model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
	return model


def train_cnnlstm(
	x_train: np.ndarray,
	x_test: np.ndarray,
	y_train: np.ndarray,
	y_test: np.ndarray,
) -> Sequential:
	"""Train and save the CNN-LSTM deep learning model."""

	x_train_reshaped = reshape_for_dl(x_train)
	x_test_reshaped = reshape_for_dl(x_test)

	model = build_cnn_lstm_model((x_train_reshaped.shape[1], x_train_reshaped.shape[2]))

	early_stop = EarlyStopping(
		monitor="val_loss",
		patience=5,
		restore_best_weights=True,
	)
	checkpoint = ModelCheckpoint(
		CNN_LSTM_MODEL_PATH,
		save_best_only=True,
		monitor="val_accuracy",
		mode="max",
	)

	model.fit(
		x_train_reshaped,
		y_train,
		epochs=50,
		batch_size=64,
		validation_data=(x_test_reshaped, y_test),
		callbacks=[early_stop, checkpoint],
		verbose=1,
	)

	evaluation_loss, evaluation_accuracy = model.evaluate(x_test_reshaped, y_test, verbose=0)
	print(f"CNN-LSTM Accuracy: {evaluation_accuracy:.4f}")

	model.save(CNN_LSTM_MODEL_PATH)
	return model


def run_pipeline() -> None:
	"""Run the full ML training pipeline end to end."""

	raw_data = load_data()
	features, target, _, _ = preprocess_data(raw_data)

	x_train, x_test, y_train, y_test = train_test_split(
		features,
		target,
		test_size=0.2,
		random_state=42,
		stratify=target,
	)

	train_rf(x_train, x_test, y_train, y_test)
	train_cnnlstm(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
	run_pipeline()
