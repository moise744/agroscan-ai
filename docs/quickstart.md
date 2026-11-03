# AgroScan AI — Quick Start Guide

Get up and running in under 5 minutes.

## Prerequisites

- Python 3.9+
- pip

## 1. Clone and Install

```bash
git clone https://github.com/moise744/agroscan-ai.git
cd agroscan-ai
pip install -e ".[dev]"
```

## 2. Verify Installation

```bash
python -m pytest tests/ -q
# Expected: 21 passed
```

## 3. Run the Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
# Opens at http://localhost:8504
```

Upload a leaf image. The system predicts the disease class, shows confidence scores, and displays a Grad-CAM saliency overlay.

## 4. Run the FastAPI Server

```bash
pip install fastapi uvicorn[standard]
uvicorn app.api:app --host 0.0.0.0 --port 8003 --reload
# API docs at http://localhost:8003/docs
```

## 5. Diagnose a Single Image

```bash
python scripts/predict.py --image path/to/leaf.jpg
```

## 6. Train on PlantVillage

```bash
# Download PlantVillage first (see data/README.md)
python scripts/train.py --config configs/classifier.yaml

# Evaluate
python scripts/evaluate.py --checkpoint outputs/best_model.pth
```

## Key Commands Reference

| Command | Purpose |
|---------|---------|
| `make install` | Install production deps |
| `make test` | Run all tests |
| `make run-app` | Launch Streamlit dashboard |
| `make run-api` | Launch FastAPI server |
| `make docker-build` | Build Docker image |
