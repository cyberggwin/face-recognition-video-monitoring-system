# Real-Time Face Recognition Monitoring System

Diploma project: an end-to-end video monitoring system that detects people in a live camera stream, identifies known faces, validates liveness, and exposes the result to Windows and Android clients.

The project was designed to run locally on a laptop, without requiring cloud inference for the live recognition pipeline.

## Highlights

- Real-time person detection using YOLOv8.
- Face recognition based on 128-dimensional facial embeddings.
- Liveness verification using facial landmarks, blink detection and head movement cues.
- Windows desktop client for live monitoring, server control, logs, notifications and local recording.
- Android client for live viewing, event history and push notifications.
- Google Drive dataset synchronization for updating known persons.
- Benchmark scripts for open-set evaluation, threshold sweep, PCA/LDA visualization and Precision/Recall/F1 metrics.

## Demo

The `demo/` folder contains three short demonstration videos:

- `demo_1_windows_live.mp4`: live Windows monitoring flow.
- `demo_2_detection_flow.mp4`: detection and recognition flow.
- `demo_3_liveness_or_mobile.mp4`: liveness/mobile-related demonstration.

## Documentation

The final diploma documentation is available in:

```text
docs/documentatie_licenta.pdf
```

## System Architecture

```text
Google Drive dataset
        |
        v
Local dataset synchronization
        |
        v
Face embedding enrollment -> models/known_faces.pkl
        |
        v
Live camera stream
        |
        v
YOLO person detection -> face localization -> face embedding
        |
        v
Identity matching + liveness validation
        |
        v
API server -> Windows client / Android client / notifications
```

## Tech Stack

### Backend and Computer Vision

- Python 3.11
- OpenCV
- Ultralytics YOLOv8
- face_recognition / dlib
- NumPy
- Matplotlib
- Google Drive API
- Firebase Cloud Messaging integration

### Windows Client

- Tkinter
- Pillow / ImageTk
- OpenCV image processing

### Android Client

- Kotlin
- Jetpack Compose
- Material 3
- OkHttp
- Firebase Cloud Messaging

## Repository Structure

```text
src/                  Python backend, computer vision and evaluation scripts
android/FaceClient/   Android client project
docs/                 Final diploma documentation PDF
demo/                 Demonstration videos
reports/              Benchmark plots and CSV results
models/               Local model/output folder placeholder
dataset/              Local dataset placeholder
```

## Privacy and Security Notes

This public portfolio version intentionally does not include:

- private Google credentials;
- Firebase service account keys;
- Android `google-services.json`;
- OAuth tokens;
- personal face dataset;
- generated facial embedding database;
- virtual environments or build artifacts.

To run the full system, add your own credentials and dataset locally.

## Quick Start

Run the commands from the repository root.

### 1. Create the Python environment

```powershell
py -3.11 -m venv venv311
.\venv311\Scripts\python.exe -m pip install --upgrade pip
.\venv311\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt
```

### 2. Prepare the dataset

Expected local dataset structure:

```text
dataset/
├─ Person_1/
│  ├─ image_1.jpg
│  └─ image_2.jpg
├─ Person_2/
│  └─ image_1.jpg
```

Each folder name is used as the label for that person.

### 3. Generate known face embeddings

```powershell
.\venv311\Scripts\python.exe .\src\enroll_faces.py --dataset .\dataset --output .\models\known_faces.pkl --jitters 2
```

### 4. Start the live server

```powershell
.\Start-Server.ps1
```

The server exposes live snapshots and events for the desktop and Android clients.

### 5. Start the Windows client

```powershell
.\venv311\Scripts\python.exe .\src\face_client_windows.py
```

### 6. Open the Android client

Open `android/FaceClient` in Android Studio, build the app, then configure the server URL:

```text
http://<PC_IP_ADDRESS>:8000
```

For push notifications, add your own Firebase configuration locally.

The public repository copy does not include Firebase configuration files. If push notifications are required, configure a Firebase project locally and add the Android `google-services.json` file in the app module.

## Evaluation

The project includes scripts for experimental evaluation.

### Mixed benchmark

```powershell
.\venv311\Scripts\python.exe .\src\mixed_benchmark.py --encodings .\models\known_faces.pkl --out-plot .\reports\mixed_benchmark.png --out-metrics-plot .\reports\metrics_benchmark.pdf
```

### Threshold sweep

```powershell
.\venv311\Scripts\python.exe .\src\threshold_sweep.py --mixed-dir .\mixed_faces --encodings .\models\known_faces.pkl --out-csv .\reports\threshold_sweep_metrics.csv --out-plot .\reports\threshold_sweep_metrics.pdf
```

### PCA / LDA embedding visualization

```powershell
.\venv311\Scripts\python.exe .\src\plot_face_embeddings_2d.py --encodings .\models\known_faces.pkl --people eu mama tata vali AMi --out .\reports\embedding_pca_lda_5_persoane.pdf
```

## Reported Results

The experimental evaluation includes:

- mixed known/unknown benchmark;
- threshold sweep for the face distance threshold;
- Precision, Recall and F1 Score calculation;
- 2D PCA and LDA visualization of facial embeddings.

For the selected threshold around `0.58`, the system achieved high recall for known identities and a good F1 score, while the main challenge remains reducing false positives in open-set recognition scenarios.

## Main Contribution

This project is not only a standalone face recognition script. It integrates:

- dataset synchronization;
- face embedding generation;
- real-time video processing;
- identity matching;
- liveness validation;
- desktop monitoring;
- mobile monitoring;
- notifications;
- benchmark and visualization tools.

The result is a practical prototype for local intelligent video monitoring.
