# Contributing

Thanks for your interest in contributing! This project is a research-oriented retrieval system, so contributions that improve evaluation rigor, retrieval quality, or code clarity are especially welcome.

## Getting Started

```bash
# Clone and set up
git clone https://github.com/ayushmath07/Semantic-Document-Retrieval-API.git
cd Semantic-Document-Retrieval-API

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

## Development Workflow

### Run Tests

```bash
pytest tests/ -v
```

All tests must pass before submitting a PR.

### Lint and Format

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check .          # lint
ruff format --check . # verify formatting
ruff format .         # auto-format
```

### Build the Index

```bash
python scripts/build_index.py --dataset scifact
```

### Run Evaluation

```bash
python scripts/evaluate.py
```

## Project Structure

```
app/              Core retrieval modules (calibration, fusion, reranking, API)
scripts/          Index building and evaluation runners
tests/            pytest test suite
eval/             Golden set labels and generated results
data/             Sample documents and FAISS/BM25 index artifacts
```

## Pull Request Guidelines

1. **One concern per PR.** Don't mix refactors with feature additions.
2. **Include tests** for new functionality.
3. **Run the full test suite** before opening the PR.
4. **Update the README** if your change affects the API, retrieval modes, or evaluation.

## Code Style

- Python 3.11+
- Type hints on all function signatures
- Docstrings on all public functions (Google style)
- Ruff-clean (`ruff check .` produces zero warnings)

## Reporting Issues

Open a GitHub issue with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Python version and OS
