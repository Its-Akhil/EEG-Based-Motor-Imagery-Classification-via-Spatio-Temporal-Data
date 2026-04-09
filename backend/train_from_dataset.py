import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

import mne
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from backend.eeg_pipeline import CHANNELS, CLASSES, Preprocessor, SAMPLING_RATE, WINDOW_SAMPLES, compute_features


EVENT_LABEL_MAP = {
    "769": "left_hand",
    "770": "right_hand",
    "771": "feet",
    "772": "tongue",
    "T1": "left_hand",
    "T2": "right_hand",
    "T3": "feet",
    "T4": "tongue",
}

CANONICAL_EXPANDED_CHANNELS = ["F3", "F4", "C3", "C4", "Cz", "P3", "P4", "Oz"]
DEFAULT_REQUIRED_CLASSES = ["left_hand", "right_hand", "feet", "tongue"]


def _normalize_channel_name(name: str) -> str:
    upper = name.upper().replace("EEG", "")
    for char in ["-", " ", "_", "."]:
        upper = upper.replace(char, "")
    return upper


def resolve_channel_order(ch_names: List[str]) -> List[str] | None:
    normalized = {_normalize_channel_name(name): name for name in ch_names}
    selected: List[str] = []

    for target in CHANNELS:
        key = _normalize_channel_name(target)
        if key not in normalized:
            return None
        selected.append(normalized[key])

    return selected


def resolve_core_motor_channels(ch_names: List[str]) -> Dict[str, str] | None:
    resolved: Dict[str, str] = {}
    normalized_to_original = {_normalize_channel_name(name): name for name in ch_names}

    for target in ["C3", "CZ", "C4"]:
        for normalized_name, original_name in normalized_to_original.items():
            if normalized_name.endswith(target):
                resolved[target] = original_name
                break

    if len(resolved) != 3:
        return None

    return resolved


def expand_to_mvp_channel_layout(epoch_core: np.ndarray) -> np.ndarray:
    c3 = epoch_core[0]
    cz = epoch_core[1]
    c4 = epoch_core[2]

    expanded = np.zeros((len(CANONICAL_EXPANDED_CHANNELS), epoch_core.shape[1]), dtype=np.float32)
    expanded[0] = 0.85 * c3  # F3 proxy
    expanded[1] = 0.85 * c4  # F4 proxy
    expanded[2] = c3
    expanded[3] = c4
    expanded[4] = cz
    expanded[5] = 0.75 * c3  # P3 proxy
    expanded[6] = 0.75 * c4  # P4 proxy
    expanded[7] = 0.70 * cz  # Oz proxy
    return expanded


def get_file_available_labels(file_path: Path) -> Set[str]:
    raw = mne.io.read_raw_gdf(file_path, preload=False, verbose="ERROR")
    _, event_id = mne.events_from_annotations(raw, verbose="ERROR")

    available: Set[str] = set()
    for annotation_desc in event_id.keys():
        label = EVENT_LABEL_MAP.get(str(annotation_desc))
        if label:
            available.add(label)
    return available


def select_training_files(
    dataset_dir: Path,
    required_classes: List[str],
    max_files: int | None = None,
    require_all_classes_in_file: bool = True,
) -> Tuple[List[Path], Dict[str, Dict[str, object]]]:
    required = set(required_classes)
    all_t_files = sorted(dataset_dir.glob("*T.gdf"))

    selected: List[Path] = []
    skipped: Dict[str, Dict[str, object]] = {}

    for file_path in all_t_files:
        available = get_file_available_labels(file_path)
        if not available:
            skipped[file_path.name] = {"reason": "no_supported_cue_events", "availableLabels": []}
            continue

        if require_all_classes_in_file and not required.issubset(available):
            skipped[file_path.name] = {
                "reason": "missing_required_classes",
                "availableLabels": sorted(available),
            }
            continue

        if not required.intersection(available):
            skipped[file_path.name] = {
                "reason": "no_required_classes_present",
                "availableLabels": sorted(available),
            }
            continue

        selected.append(file_path)

    if max_files is not None:
        selected = selected[:max_files]

    return selected, skipped


def extract_features_from_file(
    file_path: Path,
    preprocessor: Preprocessor,
    allowed_labels: Set[str],
) -> Tuple[np.ndarray, np.ndarray]:
    raw = mne.io.read_raw_gdf(file_path, preload=True, verbose="ERROR")
    raw.resample(SAMPLING_RATE, npad="auto", verbose="ERROR")

    selected_order = resolve_channel_order(raw.ch_names)
    use_expansion = False

    if not selected_order:
        core_map = resolve_core_motor_channels(raw.ch_names)
        if not core_map:
            return np.empty((0, 8), dtype=np.float32), np.empty((0,), dtype=object)
        selected_order = [core_map["C3"], core_map["CZ"], core_map["C4"]]
        use_expansion = True

    if not selected_order:
        return np.empty((0, 8), dtype=np.float32), np.empty((0,), dtype=object)

    raw.pick(selected_order)
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")

    target_event_codes: Dict[str, int] = {}
    for annotation_desc, numeric_code in event_id.items():
        if annotation_desc in EVENT_LABEL_MAP:
            target_event_codes[annotation_desc] = numeric_code

    if not target_event_codes:
        return np.empty((0, 8), dtype=np.float32), np.empty((0,), dtype=object)

    tmin = 0.5
    tmax = tmin + ((WINDOW_SAMPLES - 1) / SAMPLING_RATE)

    epochs = mne.Epochs(
        raw,
        events,
        event_id=target_event_codes,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose="ERROR",
    )

    if len(epochs) == 0:
        return np.empty((0, 8), dtype=np.float32), np.empty((0,), dtype=object)

    code_to_label = {code: EVENT_LABEL_MAP[desc] for desc, code in target_event_codes.items()}

    features: List[np.ndarray] = []
    labels: List[str] = []

    for idx, epoch in enumerate(epochs.get_data(copy=True)):
        if epoch.shape[1] != WINDOW_SAMPLES:
            continue

        if use_expansion:
            epoch = expand_to_mvp_channel_layout(epoch)

        cleaned = preprocessor.transform(epoch.astype(np.float32))
        feat = compute_features(cleaned)
        code = int(epochs.events[idx, 2])
        label = code_to_label.get(code)

        if label in allowed_labels:
            features.append(feat)
            labels.append(label)

    if not features:
        return np.empty((0, 8), dtype=np.float32), np.empty((0,), dtype=object)

    return np.vstack(features).astype(np.float32), np.array(labels, dtype=object)


def train_checkpoint(
    dataset_dir: Path,
    checkpoint_out: Path,
    report_out: Path,
    max_files: int | None = None,
    required_classes: List[str] | None = None,
    require_all_classes_in_file: bool = True,
) -> Dict[str, object]:
    required_classes = required_classes or DEFAULT_REQUIRED_CLASSES
    invalid = [label for label in required_classes if label not in CLASSES]
    if invalid:
        raise ValueError(f"Unknown required classes: {invalid}. Allowed classes are: {CLASSES}")

    gdf_files, skipped_files = select_training_files(
        dataset_dir=dataset_dir,
        required_classes=required_classes,
        max_files=max_files,
        require_all_classes_in_file=require_all_classes_in_file,
    )

    if not gdf_files:
        raise FileNotFoundError(
            f"No eligible training GDF files found in {dataset_dir}. "
            f"Check required classes {required_classes} and file suffix '*T.gdf'."
        )

    preprocessor = Preprocessor(SAMPLING_RATE)
    all_features: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    used_files: List[str] = []
    allowed_labels = set(required_classes)

    for file_path in gdf_files:
        file_features, file_labels = extract_features_from_file(file_path, preprocessor, allowed_labels)
        if len(file_labels) == 0:
            continue

        all_features.append(file_features)
        all_labels.append(file_labels)
        used_files.append(file_path.name)

    if not all_features:
        raise RuntimeError("No usable epochs were extracted from the provided GDF files.")

    x = np.vstack(all_features)
    y = np.concatenate(all_labels)

    class_counts = Counter(y.tolist())
    available_classes = sorted(class_counts.keys())
    if len(available_classes) < 2:
        raise RuntimeError("At least two classes are required for training.")

    label_to_index = {label: idx for idx, label in enumerate(available_classes)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    y_index = np.array([label_to_index[label] for label in y], dtype=np.int32)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y_index,
        test_size=0.2,
        random_state=42,
        stratify=y_index,
    )

    classifier = MLPClassifier(
        hidden_layer_sizes=(12,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=1200,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=30,
        validation_fraction=0.1,
    )
    classifier.fit(x_train, y_train)

    predictions = classifier.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))

    y_test_labels = np.array([index_to_label[int(idx)] for idx in y_test], dtype=object)
    pred_labels = np.array([index_to_label[int(idx)] for idx in predictions], dtype=object)

    learned_order = list(classifier.classes_)

    hidden_size = classifier.coefs_[1].shape[0]
    w2_full = np.zeros((hidden_size, len(CLASSES)), dtype=np.float32)
    b2_full = np.full((len(CLASSES),), -6.0, dtype=np.float32)

    if classifier.coefs_[1].shape[1] == 1 and len(learned_order) == 2:
        class0_label = index_to_label[int(learned_order[0])]
        class1_label = index_to_label[int(learned_order[1])]

        positive_w = classifier.coefs_[1][:, 0]
        positive_b = classifier.intercepts_[1][0]

        if class0_label in CLASSES:
            class0_idx = CLASSES.index(class0_label)
            w2_full[:, class0_idx] = -0.5 * positive_w
            b2_full[class0_idx] = -0.5 * positive_b

        if class1_label in CLASSES:
            class1_idx = CLASSES.index(class1_label)
            w2_full[:, class1_idx] = 0.5 * positive_w
            b2_full[class1_idx] = 0.5 * positive_b
    else:
        for learned_idx, learned_index in enumerate(learned_order):
            learned_label = index_to_label[int(learned_index)]
            if learned_label in CLASSES:
                target_idx = CLASSES.index(learned_label)
                w2_full[:, target_idx] = classifier.coefs_[1][:, learned_idx]
                b2_full[target_idx] = classifier.intercepts_[1][learned_idx]

    checkpoint_payload = {
        "classes": CLASSES,
        "availableClasses": available_classes,
        "w1": classifier.coefs_[0].astype(float).tolist(),
        "b1": classifier.intercepts_[0].astype(float).tolist(),
        "w2": w2_full.astype(float).tolist(),
        "b2": b2_full.astype(float).tolist(),
    }

    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_out.open("w", encoding="utf-8") as handle:
        json.dump(checkpoint_payload, handle, indent=2)

    cm = confusion_matrix(y_test_labels, pred_labels, labels=CLASSES)
    report = {
        "datasetDir": str(dataset_dir),
        "requiredClasses": required_classes,
        "requireAllClassesInFile": require_all_classes_in_file,
        "filesUsed": used_files,
        "skippedFiles": skipped_files,
        "totalFeatures": int(x.shape[0]),
        "trainSamples": int(x_train.shape[0]),
        "testSamples": int(x_test.shape[0]),
        "classCounts": {label: int(class_counts.get(label, 0)) for label in CLASSES},
        "availableClasses": available_classes,
        "testAccuracy": accuracy,
        "confusionMatrix": cm.astype(int).tolist(),
    }

    report_out.parent.mkdir(parents=True, exist_ok=True)
    with report_out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MVP checkpoint from GDF motor imagery data.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--checkpoint-out", type=Path, default=Path("backend/model_checkpoint.json"))
    parser.add_argument("--report-out", type=Path, default=Path("backend/training_report.json"))
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--required-classes",
        nargs="+",
        default=DEFAULT_REQUIRED_CLASSES,
        help="Class labels that must be used for training.",
    )
    parser.add_argument(
        "--allow-partial-class-files",
        action="store_true",
        help="Include training files even if some required classes are absent in that file.",
    )
    args = parser.parse_args()

    report = train_checkpoint(
        dataset_dir=args.dataset_dir,
        checkpoint_out=args.checkpoint_out,
        report_out=args.report_out,
        max_files=args.max_files,
        required_classes=args.required_classes,
        require_all_classes_in_file=not args.allow_partial_class_files,
    )

    print("Training completed.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
