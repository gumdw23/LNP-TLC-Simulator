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
