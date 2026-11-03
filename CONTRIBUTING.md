# Contributing to AgroScan AI

Thank you for your interest in contributing to AgroScan AI — a computer vision system for agricultural plant disease detection and analysis.

## Table of Contents
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Code Standards](#code-standards)
- [Test Requirements](#test-requirements)
- [Submitting Changes](#submitting-changes)

---

## Development Setup

### Prerequisites
- Python 3.9+
- Git

### Steps

1. **Fork the repository** on GitHub.

2. **Clone your fork:**
   ```bash
   git clone https://github.com/<your-username>/agroscan-ai.git
   cd agroscan-ai
   ```

3. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   # or
   make install-dev
   ```

5. **Verify the test suite passes before making changes:**
   ```bash
   make test
   ```

---

## Project Structure

```
agroscan-ai/
├── src/agroscan/
│   ├── data/           # PlantVillage dataset loader, transforms
│   ├── models/         # ResNet-based classifier, GradCAM module
│   ├── training/       # DiseaseClassifierTrainer
│   ├── evaluation/     # Per-class metrics, confusion matrix
│   └── inference/      # DiseasePredictor
├── app/
│   ├── streamlit_app.py  # Disease analysis web interface
│   └── api.py            # FastAPI /diagnose, /classes endpoints
├── scripts/            # Training, evaluation, inference scripts
├── tests/              # Unit tests (pytest)
├── notebooks/          # PlantVillage EDA and Grad-CAM notebooks
└── docs/               # Architecture and API documentation
```

---

## Code Standards

- **Style:** Follow [PEP 8](https://peps.python.org/pep-0008/). Maximum line length: **100 characters**.
- **Formatting:** Use `black` (`make format`) before committing.
- **Imports:** Sort with `isort` (`make format`).
- **Docstrings:** All public classes and functions **must** have Google-style docstrings.
- **Type hints:** All new functions should include type annotations.
- **No fabricated results:** Never hardcode disease confidence scores or accuracy numbers.
- **Honest severity:** Do not claim severity estimation unless dataset labels support it.

---

## Test Requirements

- All new features **must** include corresponding unit tests.
- All tests must pass: `make test`
- Model tests use randomly initialized weights (no pretrained download required).
- FastAPI tests use `pytest.importorskip("fastapi")`.

---

## Submitting Changes

1. Ensure all tests pass: `make test`
2. Run lint: `make lint`
3. Open a Pull Request with:
   - Clear description of the change.
   - Example Grad-CAM visualizations if the change affects explainability.
   - Reference to any related issue.

---

## Reporting Issues

Open a GitHub Issue with:
- A minimal reproducible example.
- Your Python version and OS.
- The full error traceback.
