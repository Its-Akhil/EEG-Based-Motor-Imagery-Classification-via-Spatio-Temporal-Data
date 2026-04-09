import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


SAMPLING_RATE = 250
CHUNK_SAMPLES = 25
WINDOW_SAMPLES = 250
CHANNELS = ["F3", "F4", "C3", "C4", "Cz", "P3", "P4", "Oz"]
CLASSES = ["left_hand", "right_hand", "feet", "tongue"]


@dataclass
class PredictionResult:
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    latency_ms: float


class EEGSimulator:
    """Synthetic EEG generator with class-conditioned motor imagery modulation."""

    def __init__(self, seed: int = 7) -> None:
        self.random = np.random.default_rng(seed)
        self.sample_index = 0
        self.target_class = "left_hand"
        self.phase_offsets = self.random.uniform(0.0, 2.0 * math.pi, len(CHANNELS))

    def set_seed(self, seed: int) -> None:
        self.random = np.random.default_rng(seed)
        self.sample_index = 0
        self.phase_offsets = self.random.uniform(0.0, 2.0 * math.pi, len(CHANNELS))

    def set_target_class(self, target_class: str) -> None:
        if target_class in CLASSES:
            self.target_class = target_class

    def generate_chunk(self) -> Tuple[np.ndarray, float]:
        t = (np.arange(CHUNK_SAMPLES) + self.sample_index) / SAMPLING_RATE
        self.sample_index += CHUNK_SAMPLES

        mu = np.sin(2.0 * math.pi * 10.0 * t)
        beta = np.sin(2.0 * math.pi * 20.0 * t)
        theta = np.sin(2.0 * math.pi * 6.0 * t)

        chunk = np.zeros((len(CHANNELS), CHUNK_SAMPLES), dtype=np.float32)

        channel_gain = {
            "F3": (0.7, 0.8),
            "F4": (0.7, 0.8),
            "C3": (1.2, 1.0),
            "C4": (1.2, 1.0),
            "Cz": (1.1, 1.1),
            "P3": (0.9, 0.9),
            "P4": (0.9, 0.9),
            "Oz": (0.6, 0.7),
        }

        class_mod = self._class_modulation()
        quality = float(np.clip(0.82 + self.random.normal(0.0, 0.06), 0.55, 0.98))

        for idx, channel in enumerate(CHANNELS):
            mu_scale, beta_scale = channel_gain[channel]
            mu_scale *= class_mod[channel][0]
            beta_scale *= class_mod[channel][1]

            drift = 2.5 * np.sin(2.0 * math.pi * 0.35 * t + self.phase_offsets[idx])
            noise_scale = 1.8 + (1.0 - quality) * 2.2
            noise = self.random.normal(0.0, noise_scale, CHUNK_SAMPLES)

            signal = (
                (11.0 * mu_scale * mu)
                + (6.0 * beta_scale * beta)
                + (2.5 * theta)
                + drift
                + noise
            )
            chunk[idx] = signal.astype(np.float32)

        return chunk, quality

    def _class_modulation(self) -> Dict[str, Tuple[float, float]]:
        mod = {channel: [1.0, 1.0] for channel in CHANNELS}

        if self.target_class == "left_hand":
            mod["C4"][0] = 0.38
            mod["C4"][1] = 0.75
            mod["P4"][0] = 0.6
        elif self.target_class == "right_hand":
            mod["C3"][0] = 0.38
            mod["C3"][1] = 0.75
            mod["P3"][0] = 0.6
        elif self.target_class == "feet":
            mod["Cz"][0] = 0.35
            mod["Cz"][1] = 1.45
            mod["P3"][1] = 1.2
            mod["P4"][1] = 1.2
        elif self.target_class == "tongue":
            mod["F3"][1] = 1.6
            mod["F4"][1] = 1.6
            mod["Cz"][1] = 1.2
            mod["C3"][0] = 0.8
            mod["C4"][0] = 0.8

        return {key: (value[0], value[1]) for key, value in mod.items()}


class Preprocessor:
    def __init__(self, fs: int = SAMPLING_RATE) -> None:
        self.fs = fs
        self.band_b, self.band_a = butter(4, [0.5 / (fs / 2), 40.0 / (fs / 2)], btype="band")
        self.notch_b, self.notch_a = iirnotch(50.0 / (fs / 2), Q=30.0)

    def transform(self, window: np.ndarray) -> np.ndarray:
        filtered = np.zeros_like(window, dtype=np.float32)
        for idx in range(window.shape[0]):
            band = filtfilt(self.band_b, self.band_a, window[idx]).astype(np.float32)
            cleaned = filtfilt(self.notch_b, self.notch_a, band).astype(np.float32)
            channel_std = np.std(cleaned) + 1e-6
            filtered[idx] = (cleaned - np.mean(cleaned)) / channel_std
        return filtered


class TinyMLPClassifier:
    """A lightweight pretrained two-layer neural network checkpoint loader."""

    def __init__(self, checkpoint_path: Path) -> None:
        with checkpoint_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        self.classes = payload["classes"]
        self.w1 = np.array(payload["w1"], dtype=np.float32)
        self.b1 = np.array(payload["b1"], dtype=np.float32)
        self.w2 = np.array(payload["w2"], dtype=np.float32)
        self.b2 = np.array(payload["b2"], dtype=np.float32)
        self.trained_classes = self._resolve_trained_classes(payload)

    def _resolve_trained_classes(self, payload: Dict[str, object]) -> List[str]:
        configured = payload.get("availableClasses")
        if isinstance(configured, list):
            filtered = [str(label) for label in configured if str(label) in self.classes]
            if filtered:
                return filtered

        inferred: List[str] = []
        for idx, label in enumerate(self.classes):
            weight_norm = float(np.linalg.norm(self.w2[:, idx]))
            bias_value = float(self.b2[idx])
            # Untrained fallback heads are intentionally near-zero weights with strong negative bias.
            if weight_norm > 1e-6 or bias_value > -5.5:
                inferred.append(label)

        return inferred if inferred else list(self.classes)

    def predict(self, features: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        hidden = np.maximum(0.0, (features @ self.w1) + self.b1)
        logits = (hidden @ self.w2) + self.b2
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / (np.sum(probs) + 1e-9)

        predicted_index = int(np.argmax(probs))
        predicted_class = self.classes[predicted_index]
        confidence = float(probs[predicted_index])
        all_probs = {label: float(probs[idx]) for idx, label in enumerate(self.classes)}
        return predicted_class, confidence, all_probs


def compute_features(window: np.ndarray, fs: int = SAMPLING_RATE) -> np.ndarray:
    freqs = np.fft.rfftfreq(window.shape[1], d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(window, axis=1)) ** 2

    channel_index = {name: idx for idx, name in enumerate(CHANNELS)}
    channels_of_interest = ["F3", "F4", "C3", "C4", "Cz", "P3", "P4"]

    def channel_bandpower(channel_name: str, low: float, high: float) -> float:
        mask = (freqs >= low) & (freqs <= high)
        idx = channel_index[channel_name]
        return float(np.mean(spectrum[idx, mask]))

    def safe_log(value: float) -> float:
        return float(np.log(value + 1e-8))

    mu = {name: channel_bandpower(name, 8.0, 12.0) for name in channels_of_interest}
    beta = {name: channel_bandpower(name, 13.0, 30.0) for name in channels_of_interest}
    theta = {name: channel_bandpower(name, 4.0, 7.0) for name in channels_of_interest}

    mu_left = (mu["C3"] + mu["P3"]) / 2.0
    mu_right = (mu["C4"] + mu["P4"]) / 2.0
    beta_left = (beta["C3"] + beta["P3"]) / 2.0
    beta_right = (beta["C4"] + beta["P4"]) / 2.0

    features: List[float] = []

    # Core log-bandpower profile across motor and frontal channels.
    for name in channels_of_interest:
        features.append(safe_log(mu[name]))
    for name in channels_of_interest:
        features.append(safe_log(beta[name]))
    for name in channels_of_interest:
        features.append(safe_log(theta[name]))

    # Lateralization and class-discriminative ratios.
    features.extend(
        [
            safe_log(mu_left) - safe_log(mu_right),
            safe_log(beta_left) - safe_log(beta_right),
            safe_log(mu["Cz"]) - 0.5 * (safe_log(mu["C3"]) + safe_log(mu["C4"])),
            safe_log(beta["Cz"]) - 0.5 * (safe_log(beta["C3"]) + safe_log(beta["C4"])),
            safe_log(beta["F3"]) + safe_log(beta["F4"]),
            safe_log(mu["F3"]) + safe_log(mu["F4"]),
            float(np.std(window[channel_index["Cz"]])),
            float(np.std(window)),
        ]
    )

    return np.array(features, dtype=np.float32)


class StreamRuntime:
    def __init__(self, checkpoint_path: Path) -> None:
        self.simulator = EEGSimulator()
        self.preprocessor = Preprocessor()
        self.classifier = TinyMLPClassifier(checkpoint_path)
        self.synthetic_centroids = self._build_synthetic_centroids()
        self.inference_mode = "model_plus_synthetic_calibration" if self.synthetic_centroids else "model_only"

        self.running = False
        self.target_class = "left_hand"
        self.sample_buffer: Deque[np.ndarray] = deque(maxlen=WINDOW_SAMPLES)
        class_count = max(1, len(self.classifier.classes))
        initial_prob = 1.0 / class_count
        self.smoothed_probabilities: Dict[str, float] = {label: initial_prob for label in self.classifier.classes}

    def _build_synthetic_centroids(self, calibration_seconds: int = 6, seed: int = 101) -> Dict[str, np.ndarray]:
        centroids: Dict[str, np.ndarray] = {}
        total_chunks = max(1, int((calibration_seconds * SAMPLING_RATE) / CHUNK_SAMPLES))

        for idx, label in enumerate(self.classifier.classes):
            simulator = EEGSimulator(seed=seed + (idx * 37))
            simulator.set_target_class(label)
            buffer: Deque[np.ndarray] = deque(maxlen=WINDOW_SAMPLES)
            features: List[np.ndarray] = []

            for _ in range(total_chunks):
                chunk, _ = simulator.generate_chunk()
                for sample_idx in range(chunk.shape[1]):
                    buffer.append(chunk[:, sample_idx])

                if len(buffer) < WINDOW_SAMPLES:
                    continue

                stacked = np.stack(list(buffer), axis=1)
                cleaned = self.preprocessor.transform(stacked)
                features.append(compute_features(cleaned))

            if features:
                centroids[label] = np.mean(np.vstack(features), axis=0).astype(np.float32)

        return centroids

    def _synthetic_calibration_probabilities(self, features: np.ndarray) -> Dict[str, float]:
        if not self.synthetic_centroids:
            return {label: 0.0 for label in self.classifier.classes}

        distances: List[float] = []
        labels: List[str] = []
        for label in self.classifier.classes:
            centroid = self.synthetic_centroids.get(label)
            if centroid is None:
                continue
            labels.append(label)
            distances.append(float(np.linalg.norm(features - centroid)))

        if not labels:
            return {label: 0.0 for label in self.classifier.classes}

        distance_array = np.array(distances, dtype=np.float32)
        calibration_logits = -distance_array / 0.35
        calibration_logits -= float(np.max(calibration_logits))
        probs = np.exp(calibration_logits)
        probs = probs / (float(np.sum(probs)) + 1e-9)

        output = {label: 0.0 for label in self.classifier.classes}
        for idx, label in enumerate(labels):
            output[label] = float(probs[idx])
        return output

    def _blended_probabilities(
        self,
        model_probabilities: Dict[str, float],
        calibration_probabilities: Dict[str, float],
    ) -> Dict[str, float]:
        if not self.synthetic_centroids:
            return model_probabilities

        blended: Dict[str, float] = {}
        for label in self.classifier.classes:
            model_p = float(model_probabilities.get(label, 0.0))
            calibration_p = float(calibration_probabilities.get(label, 0.0))
            blended[label] = (0.35 * model_p) + (0.65 * calibration_p)

        total = sum(blended.values()) + 1e-9
        return {label: value / total for label, value in blended.items()}

    def start(self, seed: int | None = None) -> None:
        self.running = True
        self.sample_buffer.clear()
        if seed is not None:
            self.simulator.set_seed(seed)
        self.simulator.set_target_class(self.target_class)

    def stop(self) -> None:
        self.running = False

    def set_target_class(self, target_class: str) -> None:
        if target_class in CLASSES:
            self.target_class = target_class
            self.simulator.set_target_class(target_class)

    def next_frame(self) -> Tuple[np.ndarray, float]:
        chunk, quality = self.simulator.generate_chunk()
        for sample_idx in range(chunk.shape[1]):
            self.sample_buffer.append(chunk[:, sample_idx])
        return chunk, quality

    def has_full_window(self) -> bool:
        return len(self.sample_buffer) >= WINDOW_SAMPLES

    def predict_latest(self) -> PredictionResult:
        if not self.has_full_window():
            return PredictionResult("unknown", 0.0, {label: 0.0 for label in self.classifier.classes}, 0.0)

        stacked = np.stack(list(self.sample_buffer), axis=1)

        start = np.datetime64("now")
        cleaned = self.preprocessor.transform(stacked)
        features = compute_features(cleaned)
        _, _, model_probabilities = self.classifier.predict(features)
        calibration_probabilities = self._synthetic_calibration_probabilities(features)
        probabilities = self._blended_probabilities(model_probabilities, calibration_probabilities)

        alpha = 0.35
        for label, prob in probabilities.items():
            current = self.smoothed_probabilities[label]
            self.smoothed_probabilities[label] = (1.0 - alpha) * current + alpha * prob

        best_label = max(self.smoothed_probabilities, key=self.smoothed_probabilities.get)
        best_conf = float(self.smoothed_probabilities[best_label])

        elapsed = (np.datetime64("now") - start) / np.timedelta64(1, "ms")

        return PredictionResult(
            predicted_class=best_label,
            confidence=best_conf,
            probabilities={k: float(v) for k, v in self.smoothed_probabilities.items()},
            latency_ms=float(elapsed),
        )
