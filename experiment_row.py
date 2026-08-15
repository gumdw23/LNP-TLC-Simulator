"""Build one real-experiment record dict from form input.

Pure aside from RDKit (SMILES validation/canonicalization) and the
ExperimentID/Date timestamps -- no Streamlit, no session_state. Split out
of app.py for the same reason as validation.py/mobile_phase.py: nothing
inside app.py's own top-level script can be imported and tested without a
running Streamlit session.
"""

import uuid
from datetime import datetime

from rdkit import Chem


def create_experiment_row(
    compound_name,
    smiles,
    mobile_phase,
    additive,
    additive_percent,
    plate,
    concentration,
    spotting_volume,
    development_distance,
    chamber_saturated,
    detection_method,
    experimental_rf,
    tailing,
    spot_quality,
    notes,
):
    """Validate and assemble one real-experiment row.

    Raises ValueError with a Korean message (surfaced directly in the UI)
    when compound_name is blank, smiles is not RDKit-parseable, or
    experimental_rf is outside [0, 1].
    """

    compound_name = str(compound_name).strip()
    smiles = str(smiles).strip()

    if not compound_name:
        raise ValueError("Compound name을 입력하세요.")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit이 읽을 수 없는 SMILES입니다.")

    experimental_rf = float(experimental_rf)
    if not (0 <= experimental_rf <= 1):
        raise ValueError("Rf는 0~1 사이여야 합니다.")

    slots = [("None", 0.0), ("None", 0.0), ("None", 0.0)]
    for index, item in enumerate(mobile_phase):
        slots[index] = item

    return {
        "ExperimentID": "EXP_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6],
        "Date": datetime.now().isoformat(),
        "CompoundName": compound_name,
        "SMILES": Chem.MolToSmiles(mol, canonical=True),
        "Solvent1": slots[0][0],
        "Solvent1Percent": slots[0][1],
        "Solvent2": slots[1][0],
        "Solvent2Percent": slots[1][1],
        "Solvent3": slots[2][0],
        "Solvent3Percent": slots[2][1],
        "Additive": str(additive),
        "AdditivePercent": float(additive_percent),
        "Plate": str(plate),
        "SampleConcentration_mg_mL": float(concentration),
        "SpottingVolume_uL": float(spotting_volume),
        "DevelopmentDistance_cm": float(development_distance),
        "ChamberSaturated": bool(chamber_saturated),
        "DetectionMethod": str(detection_method),
        "Experimental_Rf": experimental_rf,
        "Tailing": str(tailing),
        "SpotQuality": str(spot_quality),
        "Notes": str(notes),
        "DataOrigin": "RealExperiment",
    }
