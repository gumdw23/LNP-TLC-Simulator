# LNP TLC Simulator

Ionizable lipid TLC prediction and experimental learning web application.

## Main functions

- Molecular structure analysis from SMILES
- Rf prediction with applicability-domain warnings
- Binary mobile-phase optimization
- Supported ternary mobile-phase mapping
- Virtual TLC plate
- Real TLC experiment CSV upload
- Automatic experiment QC
- LNP correction-model training
- Tailing-model training
- Project-state export and import

## Scientific warning

The global model was trained primarily on public small-molecule TLC data.
Predictions for HIGH OOD ionizable lipids are exploratory until sufficient
real LNP TLC data and unseen-lipid validation are available.

## Deployment

This project is prepared for GitHub and Streamlit Community Cloud.

## Local development

```
python -m venv .venv
.venv\Scripts\activate       # Windows; use .venv/bin/activate on Linux/macOS
pip install -r requirements.txt ipython
streamlit run app.py
```

`ipython` isn't in `requirements.txt` but is required to unpickle
`checkpoint/core_runtime_bundle.dill` at startup.

## Tests

`validation.py`, `mobile_phase.py`, and `experiment_row.py` are pure-logic
modules split out of `app.py` specifically so they can be unit-tested
without a running Streamlit session (`app.py` itself runs Streamlit UI
calls at import time, which only works inside `streamlit run`).

```
pip install pytest
pytest tests/
```
