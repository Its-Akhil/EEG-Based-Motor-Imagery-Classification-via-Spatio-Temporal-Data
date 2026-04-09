# EEG Motor Imagery BCI MVP

A rapid MVP for EEG-based motor imagery classification with a Python backend and a two-screen web frontend.

## What is included

- Screen A: synthetic live EEG waveform with motor imagery class controls
- Screen B: live model output with predicted class, confidence, trend, and latency
- Signal preprocessing: bandpass + notch + normalization
- Lightweight pretrained neural model loaded from checkpoint

## Classes

- left_hand
- right_hand
- feet
- tongue

## Quick start (Windows PowerShell)

```powershell
cd "c:\Users\Nikhil Gupta\Desktop\Projects\BCI LAB"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Open: http://127.0.0.1:8000

## MVP demo flow

1. Open Screen A and click Start Stream.
2. Change the target class using the class buttons and observe waveform shifts.
3. Switch to Screen B and monitor live prediction/confidence updates.
4. Use Stop Stream to end the session.

## Train model from dataset

If you have GDF training files in the dataset folder, run:

```powershell
cd "c:\Users\Nikhil Gupta\Desktop\Projects\BCI LAB"
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m backend.train_from_dataset --dataset-dir dataset
```

This updates backend/model_checkpoint.json and writes backend/training_report.json.

After training, restart the app server so the new checkpoint is loaded.

Training behavior:

- Only files ending with T are used for training.
- Files ending with E are ignored (evaluation split).
- Default training requires files to contain all four cue classes: left_hand, right_hand, feet, tongue.
- If you want to include partial-class T files as well, add --allow-partial-class-files.

Current dataset note:

- The attached GDF set currently yields trained coverage for left_hand and right_hand.
- The app still keeps feet and tongue in the 4-class UI contract, but marks classes that were not learned from the latest training run.
- Check backend/training_report.json for availableClasses and classCounts after each run.

## Notes

- This is a local MVP intentionally optimized for speed of delivery.
- Security, production hardening, and broader test coverage are intentionally deferred.
- In synthetic stream mode, runtime inference blends the trained model output with a synthetic-domain calibration head so Screen B tracks selected motor imagery targets more reliably.
