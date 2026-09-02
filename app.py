
# ============================================================
# LNP TLC SIMULATOR
# Streamlit Web Application
# ============================================================

import os
import io
import json
import uuid
import zipfile
import tempfile
import contextlib
from pathlib import Path
from datetime import datetime

import dill
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from rdkit import Chem
from rdkit.Chem import Draw

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

from validation import group_holdout_split as _group_holdout_split
from mobile_phase import normalize_mobile_phase, mobile_phase_text
from experiment_row import create_experiment_row
from lnp_structural_domain import assess_structural_domain, SOURCES as LNP_STRUCTURAL_SOURCES
import db_client
from cloud_record_mapping import build_cloud_tlc_body
from theme import inject_theme


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="LNP TLC Simulator",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Shared with the LNP Lab Platform main Portal (theme.py's own docstring),
# so this app and the Portal read as one product family. Owner-requested
# sweeping UI update, 2026-09-03.
inject_theme()


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent

MODEL_DIR = (
    BASE_DIR
    /
    "models"
)

CHECKPOINT_DIR = (
    BASE_DIR
    /
    "checkpoint"
)

REFERENCE_DIR = (
    BASE_DIR
    /
    "data"
    /
    "reference"
)

TEMPLATE_DIR = (
    BASE_DIR
    /
    "data"
    /
    "templates"
)

RUNTIME_DATA_DIR = (
    BASE_DIR
    /
    "runtime_data"
)

RUNTIME_MODEL_DIR = (
    BASE_DIR
    /
    "runtime_models"
)


RUNTIME_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RUNTIME_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


RUNTIME_DB_FILE = (
    RUNTIME_DATA_DIR
    /
    "experiment_db.csv"
)


# ============================================================
# SCIENTIFIC CONSTANTS / UI
# ============================================================

GLOBAL_MODEL_SOLVENTS_FALLBACK = [
    "Hexane",
    "EtOAc",
    "DCM",
    "MeOH",
    "Et2O",
]


ALL_SOLVENTS_FALLBACK = [
    "Hexane",
    "Heptane",
    "Toluene",
    "Et2O",
    "DCM",
    "CHCl3",
    "THF",
    "EtOAc",
    "Acetone",
    "IPA",
    "EtOH",
    "MeOH",
    "ACN",
    "DMF",
    "DMSO",
]


VALID_TAILING = [
    "No",
    "Mild",
    "Moderate",
    "Severe",
]


VALID_SPOT_QUALITY = [
    "Good",
    "Diffuse",
    "Overloaded",
    "Streaking",
    "NotVisible",
]


EXPERIMENT_COLUMNS = [

    "ExperimentID",
    "Date",

    "CompoundName",
    "SMILES",

    "Solvent1",
    "Solvent1Percent",

    "Solvent2",
    "Solvent2Percent",

    "Solvent3",
    "Solvent3Percent",

    "Additive",
    "AdditivePercent",

    "Plate",

    "SampleConcentration_mg_mL",
    "SpottingVolume_uL",
    "DevelopmentDistance_cm",

    "ChamberSaturated",
    "DetectionMethod",

    "Experimental_Rf",

    "Tailing",
    "SpotQuality",

    "Notes",

    "DataOrigin",
]



# ============================================================
# IPYTHON / COLAB DILL COMPATIBILITY SHIM
#
# core_runtime_bundle.dill was created in a Colab/IPython
# environment. Some serialized references point to
# IPython.core.display, while newer IPython versions expose
# them through IPython.display.
# ============================================================

try:
    import IPython.display as _ip_display
    import IPython.core.display as _ip_core_display

    for _ip_name in dir(_ip_display):

        if _ip_name.startswith("_"):
            continue

        if not hasattr(
            _ip_core_display,
            _ip_name
        ):

            try:
                setattr(
                    _ip_core_display,
                    _ip_name,
                    getattr(
                        _ip_display,
                        _ip_name
                    )
                )

            except Exception:
                pass

except Exception:
    pass


# ============================================================
# ENGINE LOADING
# ============================================================

@st.cache_resource
def load_engine():

    bundle_file = (
        CHECKPOINT_DIR
        /
        "core_runtime_bundle.dill"
    )

    global_model_file = (
        MODEL_DIR
        /
        "global_rf_model.pkl"
    )


    if not bundle_file.exists():

        raise FileNotFoundError(
            "core_runtime_bundle.dill을 찾을 수 없습니다."
        )


    if not global_model_file.exists():

        raise FileNotFoundError(
            "global_rf_model.pkl을 찾을 수 없습니다."
        )


    with open(
        bundle_file,
        "rb"
    ) as file:

        bundle = dill.load(
            file
        )


    global_model = joblib.load(
        global_model_file
    )


    correction_file = (
        MODEL_DIR
        /
        "lnp_correction_model.pkl"
    )


    tailing_file = (
        MODEL_DIR
        /
        "tailing_model.pkl"
    )


    correction_model = (

        joblib.load(
            correction_file
        )

        if correction_file.exists()

        else None
    )


    tailing_model = (

        joblib.load(
            tailing_file
        )

        if tailing_file.exists()

        else None
    )


    return (
        bundle,
        global_model,
        correction_model,
        tailing_model
    )


try:

    (
        ENGINE,
        BASE_GLOBAL_MODEL,
        BASE_CORRECTION_MODEL,
        BASE_TAILING_MODEL,
    ) = load_engine()


except Exception as error:

    st.error(
        "계산 엔진을 불러오지 못했습니다."
    )

    st.exception(
        error
    )

    st.stop()


# bundle 내용을 현재 module에 등록
globals().update(
    ENGINE
)


# ============================================================
# REFERENCE DATABASES
# ============================================================

def load_reference_csv(
    filename
):

    path = (
        REFERENCE_DIR
        /
        filename
    )

    if path.exists():

        try:

            return pd.read_csv(
                path
            )

        except Exception:

            return pd.DataFrame()

    return pd.DataFrame()


reference_solvent_db = (
    load_reference_csv(
        "solvent_db.csv"
    )
)


reference_additive_db = (
    load_reference_csv(
        "additive_db.csv"
    )
)


if (
    len(reference_solvent_db) > 0
    and
    "ShortName"
    in reference_solvent_db.columns
):

    UI_SOLVENTS_APP = (

        reference_solvent_db[
            "ShortName"
        ]

        .dropna()

        .astype(str)

        .unique()

        .tolist()
    )

else:

    UI_SOLVENTS_APP = (
        ALL_SOLVENTS_FALLBACK
    )


GLOBAL_MODEL_SOLVENTS_APP = (

    ENGINE.get(
        "GLOBAL_MODEL_SOLVENTS",
        GLOBAL_MODEL_SOLVENTS_FALLBACK
    )
)


# ============================================================
# SESSION STATE
# ============================================================

def load_initial_database():

    if RUNTIME_DB_FILE.exists():

        try:

            return pd.read_csv(
                RUNTIME_DB_FILE
            )

        except Exception:

            pass


    template = (
        TEMPLATE_DIR
        /
        "empty_experiment_db.csv"
    )


    if template.exists():

        try:

            data = pd.read_csv(
                template
            )

            for column in EXPERIMENT_COLUMNS:

                if column not in data.columns:

                    data[
                        column
                    ] = np.nan

            return data

        except Exception:

            pass


    return pd.DataFrame(
        columns=EXPERIMENT_COLUMNS
    )


if "experiment_db" not in st.session_state:

    st.session_state[
        "experiment_db"
    ] = load_initial_database()


if "lnp_correction_model" not in st.session_state:

    st.session_state[
        "lnp_correction_model"
    ] = BASE_CORRECTION_MODEL


if "tailing_model" not in st.session_state:

    st.session_state[
        "tailing_model"
    ] = BASE_TAILING_MODEL


if "LNP_CORRECTION_FEATURES" not in st.session_state:

    st.session_state[
        "LNP_CORRECTION_FEATURES"
    ] = ENGINE.get(
        "LNP_CORRECTION_FEATURES",
        []
    )


if "TAILING_MODEL_FEATURES" not in st.session_state:

    st.session_state[
        "TAILING_MODEL_FEATURES"
    ] = ENGINE.get(
        "TAILING_MODEL_FEATURES",
        []
    )


# ============================================================
# ENGINE GLOBAL SYNCHRONIZATION
#
# dill로 복구한 함수는 자기 __globals__를 가지고 있을 수 있으므로
# 현재 DB / 모델 상태를 명시적으로 동기화한다.
# ============================================================

def sync_engine_state():

    shared = {

        "real_rf_model":
            BASE_GLOBAL_MODEL,

        "lnp_correction_model":
            st.session_state[
                "lnp_correction_model"
            ],

        "tailing_model":
            st.session_state[
                "tailing_model"
            ],

        "web_experiment_db":
            st.session_state[
                "experiment_db"
            ],

        "WEB_EXPERIMENT_FILE":
            str(
                RUNTIME_DB_FILE
            ),

        "LNP_CORRECTION_MODEL_FILE":
            str(
                RUNTIME_MODEL_DIR
                /
                "lnp_correction_model.pkl"
            ),

        "LNP_CORRECTION_FEATURE_FILE":
            str(
                RUNTIME_MODEL_DIR
                /
                "lnp_correction_features.pkl"
            ),

        "LNP_CORRECTION_FEATURES":
            st.session_state[
                "LNP_CORRECTION_FEATURES"
            ],

        "TAILING_MODEL_FEATURES":
            st.session_state[
                "TAILING_MODEL_FEATURES"
            ],

        "GLOBAL_MODEL_SOLVENTS":
            GLOBAL_MODEL_SOLVENTS_APP,

        "UI_SOLVENTS":
            UI_SOLVENTS_APP,
    }


    globals().update(
        shared
    )


    # bundle 내부 함수의 global namespace에도 반영
    for obj in ENGINE.values():

        if (
            callable(obj)
            and
            hasattr(
                obj,
                "__globals__"
            )
        ):

            try:

                obj.__globals__.update(
                    shared
                )

            except Exception:

                pass


sync_engine_state()


# ============================================================
# DATABASE HELPERS
# ============================================================

def save_runtime_db():

    db = (
        st.session_state[
            "experiment_db"
        ]
    )

    db.to_csv(
        RUNTIME_DB_FILE,
        index=False
    )



# ============================================================
# QC HELPER
# ============================================================

def run_current_qc():

    sync_engine_state()


    db = (
        st.session_state[
            "experiment_db"
        ].copy()
    )


    if len(db) == 0:

        return db


    if "run_web_experiment_qc" not in globals():

        raise RuntimeError(
            "QC engine을 찾을 수 없습니다."
        )


    qc_db = (
        run_web_experiment_qc(
            dataframe=db,
            save=False
        )
    )


    st.session_state[
        "experiment_db"
    ] = qc_db.copy()


    save_runtime_db()

    sync_engine_state()


    return qc_db


# ============================================================
# DATASET PREPARATION
# ============================================================

def prepare_correction_dataset():

    sync_engine_state()


    if (
        "build_real_lnp_ml_dataset"
        not in globals()
    ):

        raise RuntimeError(
            "ML dataset engine을 찾을 수 없습니다."
        )


    data = (
        build_real_lnp_ml_dataset(

            dataframe=(
                st.session_state[
                    "experiment_db"
                ]
            ),

            correction_only=True
        )
    )


    return data


def prepare_rf_dataset():

    sync_engine_state()


    return build_real_lnp_ml_dataset(

        dataframe=(
            st.session_state[
                "experiment_db"
            ]
        ),

        correction_only=False
    )


def prepare_tailing_dataset():

    sync_engine_state()


    # 기존 bundle function이 있으면 우선 사용
    if (
        "build_real_tailing_training_dataset"
        in globals()
    ):

        function = (
            build_real_tailing_training_dataset
        )


        try:

            function.__globals__[
                "web_experiment_db"
            ] = (
                st.session_state[
                    "experiment_db"
                ]
            )

        except Exception:

            pass


        try:

            data = function()

            return data

        except Exception:

            pass


    # fallback
    data = prepare_rf_dataset()


    if len(data) == 0:

        return pd.DataFrame()


    if (
        "UseForTailingTraining"
        in data.columns
    ):

        data = data[
            data[
                "UseForTailingTraining"
            ] == True
        ].copy()


    if (
        "ModerateOrWorse"
        not in data.columns
    ):

        return pd.DataFrame()


    data[
        "ModerateOrWorse"
    ] = pd.to_numeric(

        data[
            "ModerateOrWorse"
        ],

        errors="coerce"
    )


    data = data.dropna(

        subset=[
            "ModerateOrWorse",
            "InputSMILES",
        ]
    )


    data[
        "ModerateOrWorse"
    ] = (
        data[
            "ModerateOrWorse"
        ]
        .astype(int)
    )


    return data.reset_index(
        drop=True
    )


# ============================================================
# LOCAL MODEL TRAINING
#
# Streamlit 서버의 /content 경로에 의존하지 않는다.
# ============================================================
#
# 검증(validation) 정책
# ----------------------
# 이전 버전은 train/test 분리 없이 전체 데이터로 fit()만 하고
# "ACTIVE BUT UNVALIDATED"라는 문구만 영원히 띄웠다 — 재학습을 몇 번을
# 하든 실제로 얼마나 잘 맞는지 알 방법이 코드 안에 전혀 없었다.
#
# 여기서는 같은 지질(lipid, InputSMILES 기준)의 행이 학습 세트와 검증
# 세트에 동시에 들어가지 않도록 그룹 단위로 나눈다 — 무작위로 행 단위로만
# 나누면 같은 지질의 다른 스팟이 양쪽에 섞여 실제보다 훨씬 좋아 보이는
# 성능(leakage)이 나올 수 있기 때문이다. "unseen lipid validation"이라는
# 문구가 이미 이 프로젝트 화면에 있었으니, 이번 수정은 그 문구를 실제로
# 지키는 코드를 추가한 것이다.
#
# 검증에 쓴 분리는 성능을 "재는" 용도로만 쓰고, 실제 배포하는 모델은
# 이후 전체 데이터로 다시 학습한다(표준적인 방식) — 검증 세트만큼 데이터를
# 버리고 배포할 이유가 없다.


def train_correction_model_app():

    data = (
        prepare_correction_dataset()
    )


    n_rows = len(
        data
    )


    n_lipids = (

        data[
            "InputSMILES"
        ].nunique()

        if (
            n_rows > 0
            and
            "InputSMILES"
            in data.columns
        )

        else 0
    )


    # 프로젝트 safety gate
    if n_rows < 20:

        raise ValueError(
            f"Correction 학습 데이터가 부족합니다: "
            f"{n_rows}/20 rows"
        )


    if n_lipids < 5:

        raise ValueError(
            f"서로 다른 lipid가 부족합니다: "
            f"{n_lipids}/5"
        )


    candidates = ENGINE.get(

        "CORRECTION_FEATURE_CANDIDATES",

        globals().get(
            "CORRECTION_FEATURE_CANDIDATES",
            []
        )
    )


    features = [

        feature

        for feature in candidates

        if feature
        in data.columns
    ]


    if (
        "GlobalPredicted_Rf"
        not in features
    ):

        raise ValueError(
            "GlobalPredicted_Rf feature가 없습니다."
        )


    if len(features) < 5:

        raise ValueError(
            "학습 가능한 correction feature가 너무 적습니다."
        )


    X = (

        data[
            features
        ]

        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )


    y = pd.to_numeric(

        data[
            "RfResidual"
        ],

        errors="coerce"
    )


    valid = y.notna()


    X = X.loc[
        valid
    ]


    y = y.loc[
        valid
    ]


    groups = data.loc[X.index, "InputSMILES"]


    def _make_pipeline():
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ])


    # held-out 지질(lipid)로 정직하게 성능을 재고, 배포 모델은 따로
    # 전체 데이터로 다시 학습한다 (아래 "held-out 검증" 절 참고).
    train_idx, test_idx, split_warning = _group_holdout_split(
        len(X), groups.to_numpy(), min_test_groups=1
    )

    metrics = None
    if len(test_idx) > 0:
        eval_model = _make_pipeline()
        eval_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = eval_model.predict(X.iloc[test_idx])
        metrics = {
            "r2": r2_score(y.iloc[test_idx], y_pred),
            "rmse": mean_squared_error(y.iloc[test_idx], y_pred) ** 0.5,
            "mae": mean_absolute_error(y.iloc[test_idx], y_pred),
            "n_test_rows": int(len(test_idx)),
            "n_test_lipids": int(groups.iloc[test_idx].nunique()),
        }


    model = _make_pipeline()


    model.fit(
        X,
        y
    )


    st.session_state[
        "lnp_correction_model"
    ] = model


    st.session_state[
        "LNP_CORRECTION_FEATURES"
    ] = features


    joblib.dump(

        model,

        RUNTIME_MODEL_DIR
        /
        "lnp_correction_model.pkl"
    )


    joblib.dump(

        features,

        RUNTIME_MODEL_DIR
        /
        "lnp_correction_features.pkl"
    )


    sync_engine_state()


    return (
        model,
        data,
        n_rows,
        n_lipids,
        metrics,
        split_warning
    )


def train_tailing_model_app():

    data = (
        prepare_tailing_dataset()
    )


    n_rows = len(
        data
    )


    n_lipids = (

        data[
            "InputSMILES"
        ].nunique()

        if (
            n_rows > 0
            and
            "InputSMILES"
            in data.columns
        )

        else 0
    )


    if n_rows < 30:

        raise ValueError(
            f"Tailing 학습 데이터가 부족합니다: "
            f"{n_rows}/30 rows"
        )


    if n_lipids < 5:

        raise ValueError(
            f"서로 다른 lipid가 부족합니다: "
            f"{n_lipids}/5"
        )


    y = (
        data[
            "ModerateOrWorse"
        ]
        .astype(int)
    )


    class_counts = (
        y.value_counts()
    )


    class0 = int(
        class_counts.get(
            0,
            0
        )
    )


    class1 = int(
        class_counts.get(
            1,
            0
        )
    )


    if (
        class0 < 5
        or
        class1 < 5
    ):

        raise ValueError(
            "각 tailing class가 최소 5개 필요합니다. "
            f"No/Mild={class0}, "
            f"Moderate/Severe={class1}"
        )


    candidate_features = (

        st.session_state[
            "TAILING_MODEL_FEATURES"
        ]

        or ENGINE.get(
            "TAILING_MODEL_FEATURES",
            []
        )
    )


    features = [

        feature

        for feature
        in candidate_features

        if feature
        in data.columns
    ]


    if len(features) < 10:

        raise ValueError(
            "학습 가능한 tailing feature가 너무 적습니다."
        )


    X = (

        data[
            features
        ]

        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )


    groups = data.loc[X.index, "InputSMILES"]


    def _make_pipeline():
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42),
            ),
        ])


    # 회귀(correction) 쪽과 같은 이유로 그룹(지질) 단위 held-out 분리.
    # 분류는 클래스별로도 검증 세트에 최소 인원이 있어야 지표가 의미
    # 있으므로, 분리 후 양쪽 클래스가 모두 있는지 추가로 확인한다.
    train_idx, test_idx, split_warning = _group_holdout_split(
        len(X), groups.to_numpy(), min_test_groups=1
    )

    metrics = None
    if len(test_idx) > 0 and y.iloc[test_idx].nunique() < 2:
        split_warning = (
            "검증 세트에 한쪽 class만 남아 held-out 검증을 신뢰할 수 없습니다."
        )
        test_idx = np.array([], dtype=int)

    if len(test_idx) > 0:
        eval_model = _make_pipeline()
        eval_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = eval_model.predict(X.iloc[test_idx])
        metrics = {
            "accuracy": accuracy_score(y.iloc[test_idx], y_pred),
            "balanced_accuracy": balanced_accuracy_score(y.iloc[test_idx], y_pred),
            "f1": f1_score(y.iloc[test_idx], y_pred, zero_division=0),
            "n_test_rows": int(len(test_idx)),
            "n_test_lipids": int(groups.iloc[test_idx].nunique()),
        }


    model = _make_pipeline()


    model.fit(
        X,
        y
    )


    st.session_state[
        "tailing_model"
    ] = model


    st.session_state[
        "TAILING_MODEL_FEATURES"
    ] = features


    joblib.dump(

        model,

        RUNTIME_MODEL_DIR
        /
        "tailing_model.pkl"
    )


    joblib.dump(

        features,

        RUNTIME_MODEL_DIR
        /
        "tailing_features.pkl"
    )


    sync_engine_state()


    return (
        model,
        data,
        n_rows,
        n_lipids,
        class0,
        class1,
        metrics,
        split_warning
    )


# ============================================================
# PROJECT STATE ZIP
#
# Community Cloud runtime disk is not permanent.
# DB + trained model을 사용자가 직접 백업/복구할 수 있게 한다.
# ============================================================

def build_state_zip():

    sync_engine_state()


    memory = io.BytesIO()


    with zipfile.ZipFile(
        memory,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as archive:


        # DB
        db_csv = (
            st.session_state[
                "experiment_db"
            ]
            .to_csv(
                index=False
            )
        )


        archive.writestr(
            "experiment_db.csv",
            db_csv
        )


        # metadata
        metadata = {

            "created_at":
                datetime.now().isoformat(),

            "experiment_rows":
                len(
                    st.session_state[
                        "experiment_db"
                    ]
                ),

            "correction_model_active":
                (
                    st.session_state[
                        "lnp_correction_model"
                    ]
                    is not None
                ),

            "tailing_model_active":
                (
                    st.session_state[
                        "tailing_model"
                    ]
                    is not None
                ),

            "warning":
                (
                    "LNP correction/tailing models require "
                    "independent unseen-lipid validation."
                ),
        }


        archive.writestr(

            "state_metadata.json",

            json.dumps(
                metadata,
                indent=2,
                ensure_ascii=False
            )
        )


        # Models
        with tempfile.TemporaryDirectory() as temp_dir:

            temp = Path(
                temp_dir
            )


            correction_model = (
                st.session_state[
                    "lnp_correction_model"
                ]
            )


            if correction_model is not None:

                correction_path = (
                    temp
                    /
                    "lnp_correction_model.pkl"
                )

                features_path = (
                    temp
                    /
                    "lnp_correction_features.pkl"
                )


                joblib.dump(
                    correction_model,
                    correction_path
                )


                joblib.dump(

                    st.session_state[
                        "LNP_CORRECTION_FEATURES"
                    ],

                    features_path
                )


                archive.write(
                    correction_path,
                    "lnp_correction_model.pkl"
                )


                archive.write(
                    features_path,
                    "lnp_correction_features.pkl"
                )


            tailing_model = (
                st.session_state[
                    "tailing_model"
                ]
            )


            if tailing_model is not None:

                tailing_path = (
                    temp
                    /
                    "tailing_model.pkl"
                )

                features_path = (
                    temp
                    /
                    "tailing_features.pkl"
                )


                joblib.dump(
                    tailing_model,
                    tailing_path
                )


                joblib.dump(

                    st.session_state[
                        "TAILING_MODEL_FEATURES"
                    ],

                    features_path
                )


                archive.write(
                    tailing_path,
                    "tailing_model.pkl"
                )


                archive.write(
                    features_path,
                    "tailing_features.pkl"
                )


    memory.seek(
        0
    )


    return memory.getvalue()


def restore_state_zip(
    uploaded_file
):

    file_bytes = (
        uploaded_file.getvalue()
    )


    with tempfile.TemporaryDirectory() as temp_dir:

        temp = Path(
            temp_dir
        )


        with zipfile.ZipFile(
            io.BytesIO(
                file_bytes
            ),
            "r"
        ) as archive:

            archive.extractall(
                temp
            )


        database_file = (
            temp
            /
            "experiment_db.csv"
        )


        if database_file.exists():

            db = pd.read_csv(
                database_file
            )

            st.session_state[
                "experiment_db"
            ] = db


        correction_file = (
            temp
            /
            "lnp_correction_model.pkl"
        )


        correction_features = (
            temp
            /
            "lnp_correction_features.pkl"
        )


        if correction_file.exists():

            st.session_state[
                "lnp_correction_model"
            ] = joblib.load(
                correction_file
            )


            if correction_features.exists():

                st.session_state[
                    "LNP_CORRECTION_FEATURES"
                ] = joblib.load(
                    correction_features
                )


        tailing_file = (
            temp
            /
            "tailing_model.pkl"
        )


        tailing_features = (
            temp
            /
            "tailing_features.pkl"
        )


        if tailing_file.exists():

            st.session_state[
                "tailing_model"
            ] = joblib.load(
                tailing_file
            )


            if tailing_features.exists():

                st.session_state[
                    "TAILING_MODEL_FEATURES"
                ] = joblib.load(
                    tailing_features
                )


    save_runtime_db()

    sync_engine_state()


# ============================================================
# COMMON MOBILE PHASE UI
# ============================================================

def mobile_phase_inputs(
    key_prefix,
    default1="Hexane",
    ratio1=70.0,
    default2="EtOAc",
    ratio2=30.0
):

    c1, c2, c3 = st.columns(
        3
    )


    with c1:

        solvent1 = st.selectbox(

            "Solvent 1",

            UI_SOLVENTS_APP,

            index=(
                UI_SOLVENTS_APP.index(
                    default1
                )
                if default1
                in UI_SOLVENTS_APP
                else 0
            ),

            key=f"{key_prefix}_s1"
        )


        r1 = st.number_input(

            "Ratio 1",

            min_value=0.0,

            value=float(
                ratio1
            ),

            step=1.0,

            key=f"{key_prefix}_r1"
        )


    with c2:

        options = (
            ["None"]
            +
            UI_SOLVENTS_APP
        )


        solvent2 = st.selectbox(

            "Solvent 2",

            options,

            index=(
                options.index(
                    default2
                )
                if default2
                in options
                else 0
            ),

            key=f"{key_prefix}_s2"
        )


        r2 = st.number_input(

            "Ratio 2",

            min_value=0.0,

            value=float(
                ratio2
            ),

            step=1.0,

            key=f"{key_prefix}_r2"
        )


    with c3:

        solvent3 = st.selectbox(

            "Solvent 3",

            ["None"]
            +
            UI_SOLVENTS_APP,

            index=0,

            key=f"{key_prefix}_s3"
        )


        r3 = st.number_input(

            "Ratio 3",

            min_value=0.0,

            value=0.0,

            step=1.0,

            key=f"{key_prefix}_r3"
        )


    return (
        solvent1,
        r1,
        solvent2,
        r2,
        solvent3,
        r3
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🧪 LNP TLC"
)


st.sidebar.caption(
    "Ionizable lipid TLC prediction & learning"
)


st.sidebar.divider()


with st.sidebar.expander(
    "☁️ 공유 데이터베이스 로그인 (선택)",
    expanded=False
):

    st.caption(
        "로그인하면 'Record Experiment'에서 저장한 실험이 이 컴퓨터의 로컬 파일뿐 아니라 "
        "연구실 공유 데이터베이스(Lab Portal과 동일)에도 함께 저장됩니다. 로그인하지 않아도 "
        "지금까지처럼 로컬 저장은 그대로 작동합니다."
    )

    if st.session_state.get("db_access_token"):

        st.success(
            f"로그인됨: {st.session_state.get('db_email', '')}"
        )

        if st.button(
            "로그아웃",
            key="db_logout_button"
        ):
            for _k in ("db_access_token", "db_email"):
                st.session_state.pop(_k, None)
            st.rerun()

    else:

        with st.form("db_login_form"):

            db_login_email = st.text_input(
                "이메일",
                key="db_login_email"
            )

            db_login_password = st.text_input(
                "비밀번호",
                type="password",
                key="db_login_password"
            )

            db_login_submitted = st.form_submit_button(
                "로그인"
            )

        if db_login_submitted:

            try:

                _login_result = db_client.sign_in(
                    db_login_email,
                    db_login_password
                )

                st.session_state["db_access_token"] = _login_result["access_token"]
                st.session_state["db_email"] = _login_result["user"]["email"]
                st.rerun()

            except db_client.ApiError as error:

                st.error(
                    f"로그인 실패: {db_client.friendly_error(error)}"
                )


st.sidebar.divider()


st.sidebar.subheader(
    "💾 Project State"
)


st.sidebar.caption(
    "실험 DB와 학습된 모델은 State ZIP으로 "
    "백업해 두는 것을 권장합니다."
)


state_zip = build_state_zip()


st.sidebar.download_button(

    "⬇️ Download State ZIP",

    data=state_zip,

    file_name=(
        "LNP_TLC_STATE_"
        +
        datetime.now().strftime(
            "%Y%m%d"
        )
        +
        ".zip"
    ),

    mime="application/zip",
)


uploaded_state = (
    st.sidebar.file_uploader(

        "⬆️ Restore State ZIP",

        type=[
            "zip"
        ],

        key="state_upload"
    )
)


if uploaded_state is not None:

    if st.sidebar.button(
        "Restore uploaded state"
    ):

        try:

            restore_state_zip(
                uploaded_state
            )

            st.sidebar.success(
                "State 복구 완료"
            )

            st.rerun()

        except Exception as error:

            st.sidebar.error(
                "State 복구 실패"
            )

            st.sidebar.exception(
                error
            )


st.sidebar.divider()


st.sidebar.warning(
    "HIGH OOD ionizable lipid 예측은 "
    "실제 LNP 데이터로 검증되기 전까지 "
    "exploratory prediction입니다."
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "🧪 LNP TLC Simulator"
)


st.caption(
    "TLC prediction · mobile-phase optimization · "
    "real-experiment learning"
)


# ============================================================
# TABS
# ============================================================

(
    tab_analyze,
    tab_optimizer,
    tab_ternary,
    tab_virtual,
    tab_record,
    tab_database,
    tab_training,
    tab_status,

) = st.tabs([

    "🔬 Analyze",
    "🎯 Optimizer",
    "🔺 Ternary",
    "🧪 Virtual TLC",
    "🧾 Record Experiment",
    "🗂 Database / QC",
    "🧠 Model Training",
    "📊 Model Status",
])


# ============================================================
# TAB 1 — ANALYZE
# ============================================================

with tab_analyze:

    st.header(
        "🔬 TLC Prediction"
    )


    smiles = st.text_input(

        "SMILES",

        value="COc1ccccc1",

        key="analyze_smiles"
    )


    (
        s1,
        r1,
        s2,
        r2,
        s3,
        r3

    ) = mobile_phase_inputs(
        "analyze"
    )


    c1, c2, c3, c4 = st.columns(
        4
    )


    with c1:

        additive = st.selectbox(

            "Additive",

            [
                "None",
                "TEA",
                "DIPEA",
                "AcOH",
                "FA",
            ],

            key="analyze_additive"
        )


    with c2:

        additive_percent = st.number_input(

            "Additive (%)",

            min_value=0.0,

            value=0.0,

            step=0.1,

            key="analyze_additive_percent"
        )


    with c3:

        concentration = st.number_input(

            "Sample concentration (mg/mL)",

            min_value=0.01,

            value=5.0,

            key="analyze_concentration"
        )


    with c4:

        spotting_volume = st.number_input(

            "Spotting volume (µL)",

            min_value=0.01,

            value=1.0,

            key="analyze_volume"
        )


    if st.button(
        "🔬 Analyze TLC",
        type="primary"
    ):

        try:

            sync_engine_state()


            mobile_phase = (
                normalize_mobile_phase([

                    (
                        s1,
                        r1
                    ),

                    (
                        s2,
                        r2
                    ),

                    (
                        s3,
                        r3
                    ),
                ])
            )


            with contextlib.redirect_stdout(
                io.StringIO()
            ):

                result = analyze_tlc(

                    smiles=smiles,

                    mobile_phase=mobile_phase,

                    additive=additive,

                    additive_percent=float(
                        additive_percent
                    ),

                    correction_model=(
                        st.session_state[
                            "lnp_correction_model"
                        ]
                    ),

                    tailing_model=(
                        st.session_state[
                            "tailing_model"
                        ]
                    ),

                    sample_concentration_mg_ml=float(
                        concentration
                    ),

                    spotting_volume_ul=float(
                        spotting_volume
                    ),

                    show_domain_details=False
                )


            mol = Chem.MolFromSmiles(
                smiles
            )


            col_structure, col_result = (
                st.columns(
                    [
                        1,
                        2
                    ]
                )
            )


            with col_structure:

                if mol is not None:

                    image = Draw.MolToImage(
                        mol,
                        size=(
                            500,
                            350
                        )
                    )

                    st.image(
                        image,
                        caption="Molecular structure"
                    )


            with col_result:

                a, b, c, d = (
                    st.columns(
                        4
                    )
                )


                a.metric(

                    "Predicted Rf",

                    (
                        f"{float(result['Predicted_Rf']):.3f}"

                        if pd.notna(
                            result.get(
                                "Predicted_Rf"
                            )
                        )

                        else "N/A"
                    )
                )


                b.metric(

                    "Reliability",

                    result.get(
                        "Reliability",
                        "N/A"
                    )
                )


                c.metric(

                    "Domain",

                    result.get(
                        "MoleculeDomain",
                        "N/A"
                    )
                )


                similarity = (
                    result.get(
                        "StructuralSimilarity",
                        np.nan
                    )
                )


                d.metric(

                    "Similarity",

                    (
                        f"{float(similarity):.3f}"

                        if pd.notna(
                            similarity
                        )

                        else "N/A"
                    )
                )


                st.write(
                    "**Mobile phase:**",
                    mobile_phase_text(
                        mobile_phase
                    )
                )


                lower = (
                    result.get(
                        "Rf90_Lower",
                        np.nan
                    )
                )


                upper = (
                    result.get(
                        "Rf90_Upper",
                        np.nan
                    )
                )


                if (
                    pd.notna(lower)
                    and
                    pd.notna(upper)
                ):

                    st.write(
                        "**90% reference interval:**",
                        f"{float(lower):.3f} – "
                        f"{float(upper):.3f}"
                    )


                st.write(
                    "**Overall status:**",
                    result.get(
                        "OverallStatus",
                        "UNKNOWN"
                    )
                )


                tailing_risk = (
                    result.get(
                        "TailingRisk",
                        np.nan
                    )
                )


                if pd.notna(
                    tailing_risk
                ):

                    st.write(
                        "**P(Moderate+ tailing):**",
                        f"{100*float(tailing_risk):.1f}%"
                    )


                warnings = (
                    result.get(
                        "Warnings",
                        []
                    )
                )


                for warning in warnings:

                    st.warning(
                        warning
                    )


            with st.expander(
                "Detailed result"
            ):

                st.json(
                    result,
                    expanded=False
                )


            st.session_state["analyze_last_smiles"] = smiles


        except Exception as error:

            st.error(
                "분석에 실패했습니다."
            )

            st.exception(
                error
            )


    st.divider()

    st.subheader(
        "🧬 LNP Structural Reference (실험적 참고용)"
    )

    st.caption(
        "이 지질 구조가 공개 LNP 문헌 라이브러리(LNPDB, LNP Atlas)에서 얼마나 낯익은 "
        "스캐폴드인지 보여주는 참고 정보입니다. 위 TLC Rf 예측의 Domain/Similarity와는 "
        "완전히 다른 지표이며, 두 라이브러리 모두 TLC Rf나 solubility 측정값을 담고 있지 "
        "않습니다 — 구조적으로 익숙하다고 해서 Rf 예측이 더 정확해지는 것은 아닙니다. "
        "첫 조회는 참조 라이브러리를 불러오느라 몇 초 걸릴 수 있어, TLC 예측과 별개로 "
        "버튼을 눌러야만 조회됩니다."
    )

    lookup_smiles = st.session_state.get(
        "analyze_last_smiles",
        smiles
    )

    if st.button(
        "🧬 구조 참조 조회",
        key="lnp_structural_lookup_button"
    ):

        for source_key, source_cfg in LNP_STRUCTURAL_SOURCES.items():

            st.session_state[
                f"lnp_domain_{source_key}"
            ] = assess_structural_domain(
                lookup_smiles,
                source_key=source_key
            )

        st.session_state["lnp_domain_smiles"] = lookup_smiles


    if st.session_state.get("lnp_domain_smiles"):

        st.caption(
            f"조회한 SMILES: {st.session_state['lnp_domain_smiles']}"
        )

        for source_key, source_cfg in LNP_STRUCTURAL_SOURCES.items():

            domain = st.session_state.get(
                f"lnp_domain_{source_key}"
            )

            if domain is None:
                continue

            st.write(f"**{source_cfg['display_name']}**")

            if not domain.valid:
                st.caption(domain.error or "평가할 수 없습니다.")
                continue

            dcol1, dcol2, dcol3 = st.columns(3)

            dcol1.metric(
                "가장 가까운 구조와의 유사도",
                f"{domain.top_similarity:.3f}"
            )

            dcol2.metric(
                "Classification (분류)",
                {
                    "in_domain": "In-domain (친숙함)",
                    "borderline": "Borderline (경계)",
                    "novel": "Novel (낯섦)",
                }.get(domain.domain_category, domain.domain_category)
            )

            dcol3.metric(
                "참조 구조 수",
                domain.reference_size
            )

            if domain.nearest_neighbors:
                nearest = domain.nearest_neighbors[0]
                st.caption(
                    f"가장 가까운 구조: {nearest.get('source_name') or '(이름 없음)'} "
                    f"(유사도 {nearest['similarity']:.3f})"
                )


# ============================================================
# TAB 2 — OPTIMIZER
# ============================================================

with tab_optimizer:

    st.header(
        "🎯 Mobile-phase Optimizer"
    )


    optimizer_smiles = st.text_input(

        "SMILES",

        value="COc1ccccc1",

        key="optimizer_smiles"
    )


    solvent_pool = st.multiselect(

        "Solvent pool",

        GLOBAL_MODEL_SOLVENTS_APP,

        default=[
            "Hexane",
            "EtOAc",
        ]
    )


    target_range = st.slider(

        "Target Rf",

        min_value=0.0,
        max_value=1.0,

        value=(
            0.20,
            0.40
        ),

        step=0.01,

        key="optimizer_target"
    )


    allow_ood = st.checkbox(

        "Allow HIGH OOD exploratory optimization",

        value=False,

        key="optimizer_ood"
    )


    if st.button(
        "🎯 Find conditions"
    ):

        try:

            if len(
                solvent_pool
            ) < 2:

                raise ValueError(
                    "Solvent를 최소 2개 선택하세요."
                )


            sync_engine_state()


            with contextlib.redirect_stdout(
                io.StringIO()
            ):

                result = (
                    recommend_tlc_conditions(

                        smiles=optimizer_smiles,

                        solvent_pool=solvent_pool,

                        target_rf=(
                            float(
                                target_range[0]
                            ),
                            float(
                                target_range[1]
                            )
                        ),

                        top_n=15,

                        allow_high_ood=allow_ood
                    )
                )


            if isinstance(
                result,
                tuple
            ):

                top_df = result[
                    0
                ]

            else:

                top_df = result


            st.success(
                "Optimizer 완료"
            )


            st.dataframe(
                top_df,
                use_container_width=True
            )


        except Exception as error:

            st.error(
                "Optimizer가 중단되었습니다."
            )

            st.exception(
                error
            )


# ============================================================
# TAB 3 — TERNARY
# ============================================================

with tab_ternary:

    st.header(
        "🔺 3-Component Mobile-phase Map"
    )


    ternary_smiles = st.text_input(

        "SMILES",

        value="COc1ccccc1",

        key="ternary_smiles"
    )


    c1, c2, c3 = st.columns(
        3
    )


    with c1:

        ta = st.selectbox(

            "Solvent A",

            GLOBAL_MODEL_SOLVENTS_APP,

            index=0,

            key="ternary_a"
        )


    with c2:

        tb = st.selectbox(

            "Solvent B",

            GLOBAL_MODEL_SOLVENTS_APP,

            index=1,

            key="ternary_b"
        )


    with c3:

        tc = st.selectbox(

            "Solvent C",

            GLOBAL_MODEL_SOLVENTS_APP,

            index=min(
                2,
                len(
                    GLOBAL_MODEL_SOLVENTS_APP
                ) - 1
            ),

            key="ternary_c"
        )


    target_ternary = st.slider(

        "Target Rf",

        0.0,
        1.0,

        (
            0.20,
            0.40
        ),

        0.01,

        key="ternary_target"
    )


    ternary_step = st.selectbox(

        "Composition step (%)",

        [
            5,
            10,
            20,
        ],

        index=0
    )


    ternary_ood = st.checkbox(

        "Allow HIGH OOD exploratory map",

        value=False,

        key="ternary_ood"
    )


    if st.button(
        "🔺 Generate ternary map"
    ):

        try:

            if len(
                {
                    ta,
                    tb,
                    tc
                }
            ) != 3:

                raise ValueError(
                    "서로 다른 3개의 solvent를 선택하세요."
                )


            sync_engine_state()


            with contextlib.redirect_stdout(
                io.StringIO()
            ):

                top_df, all_df = (
                    optimize_ternary_mobile_phase(

                        smiles=ternary_smiles,

                        solvent_a=ta,
                        solvent_b=tb,
                        solvent_c=tc,

                        target_rf=(
                            float(
                                target_ternary[0]
                            ),
                            float(
                                target_ternary[1]
                            )
                        ),

                        step=int(
                            ternary_step
                        ),

                        min_component_percent=int(
                            ternary_step
                        ),

                        correction_model=(
                            st.session_state[
                                "lnp_correction_model"
                            ]
                        ),

                        tailing_model=(
                            st.session_state[
                                "tailing_model"
                            ]
                        ),

                        allow_high_ood=ternary_ood,

                        top_n=20
                    )
                )


            figure = (
                create_ternary_rf_figure(

                    all_df,

                    target_rf=(
                        float(
                            target_ternary[0]
                        ),
                        float(
                            target_ternary[1]
                        )
                    )
                )
            )


            st.success(
                f"Supported points: {len(all_df)}"
            )


            if figure is not None:

                st.pyplot(
                    figure,
                    clear_figure=True
                )


            st.subheader(
                "Top conditions"
            )


            st.dataframe(
                top_df,
                use_container_width=True
            )


        except Exception as error:

            st.error(
                "Ternary map 생성에 실패했습니다."
            )

            st.exception(
                error
            )


# ============================================================
# TAB 4 — VIRTUAL TLC
# ============================================================

with tab_virtual:

    st.header(
        "🧪 Virtual TLC Plate"
    )


    compound_text = st.text_area(

        "Compounds — 한 줄에 `이름 | SMILES`",

        value=(
            "Anisole | COc1ccccc1\n"
            "Toluene | Cc1ccccc1\n"
            "Phenol | Oc1ccccc1"
        ),

        height=150
    )


    (
        vs1,
        vr1,
        vs2,
        vr2,
        vs3,
        vr3

    ) = mobile_phase_inputs(
        "virtual"
    )


    vc1, vc2 = st.columns(
        2
    )


    with vc1:

        virtual_additive = st.selectbox(

            "Additive",

            [
                "None",
                "TEA",
                "DIPEA",
                "AcOH",
                "FA",
            ],

            key="virtual_additive"
        )


    with vc2:

        virtual_additive_percent = (
            st.number_input(

                "Additive (%)",

                min_value=0.0,

                value=0.0,

                step=0.1,

                key="virtual_additive_percent"
            )
        )


    virtual_ood = st.checkbox(

        "Allow HIGH OOD exploratory lanes",

        value=False,

        key="virtual_ood"
    )


    if st.button(
        "🧪 Generate Virtual TLC"
    ):

        try:

            sync_engine_state()


            mobile_phase = (
                normalize_mobile_phase([

                    (
                        vs1,
                        vr1
                    ),

                    (
                        vs2,
                        vr2
                    ),

                    (
                        vs3,
                        vr3
                    ),
                ])
            )


            compounds = (
                parse_virtual_tlc_compound_text(
                    compound_text
                )
            )


            with contextlib.redirect_stdout(
                io.StringIO()
            ):

                prediction_df = (
                    predict_virtual_tlc_plate(

                        compounds=compounds,

                        mobile_phase=mobile_phase,

                        additive=virtual_additive,

                        additive_percent=float(
                            virtual_additive_percent
                        ),

                        allow_high_ood=virtual_ood
                    )
                )


            figure = (
                draw_virtual_tlc_plate(

                    prediction_df,

                    title=(
                        "Virtual TLC Plate\n"
                        +
                        mobile_phase_text(
                            mobile_phase
                        )
                    )
                )
            )


            st.pyplot(
                figure,
                clear_figure=True
            )


            st.dataframe(
                prediction_df,
                use_container_width=True
            )


        except Exception as error:

            st.error(
                "Virtual TLC 생성 실패"
            )

            st.exception(
                error
            )


# ============================================================
# TAB 5 — RECORD REAL EXPERIMENT
# ============================================================

with tab_record:

    st.header(
        "🧾 Record Real TLC Experiment"
    )


    st.info(
        "여기에는 실제로 수행한 TLC 결과만 저장하세요. "
        "예제/가상 Rf는 학습 데이터에 넣지 않습니다."
    )


    with st.form(
        "record_experiment_form"
    ):

        name = st.text_input(
            "Compound name"
        )


        record_smiles = st.text_input(
            "SMILES"
        )


        (
            rs1,
            rr1,
            rs2,
            rr2,
            rs3,
            rr3

        ) = mobile_phase_inputs(
            "record"
        )


        rc1, rc2, rc3 = (
            st.columns(
                3
            )
        )


        with rc1:

            record_additive = (
                st.selectbox(

                    "Additive",

                    [
                        "None",
                        "TEA",
                        "DIPEA",
                        "AcOH",
                        "FA",
                    ],

                    key="record_additive"
                )
            )


            record_additive_percent = (
                st.number_input(

                    "Additive (%)",

                    min_value=0.0,

                    value=0.0,

                    step=0.1,

                    key="record_additive_percent"
                )
            )


        with rc2:

            experimental_rf = (
                st.number_input(

                    "Experimental Rf",

                    min_value=0.0,

                    max_value=1.0,

                    value=0.50,

                    step=0.01
                )
            )


            tailing = st.selectbox(

                "Tailing",

                VALID_TAILING
            )


        with rc3:

            spot_quality = (
                st.selectbox(

                    "Spot quality",

                    VALID_SPOT_QUALITY
                )
            )


            plate = st.text_input(

                "Plate",

                value=(
                    "Silica gel 60 F254"
                )
            )


        ec1, ec2, ec3 = (
            st.columns(
                3
            )
        )


        with ec1:

            record_concentration = (
                st.number_input(

                    "Concentration (mg/mL)",

                    min_value=0.01,

                    value=5.0
                )
            )


        with ec2:

            record_spotting = (
                st.number_input(

                    "Spotting volume (µL)",

                    min_value=0.01,

                    value=1.0
                )
            )


        with ec3:

            development_distance = (
                st.number_input(

                    "Development distance (cm)",

                    min_value=0.1,

                    value=7.0
                )
            )


        chamber_saturated = (
            st.checkbox(

                "Chamber saturated",

                value=True
            )
        )


        detection = st.text_input(

            "Detection method",

            value="UV"
        )


        notes = st.text_area(
            "Notes"
        )


        confirm_real = st.checkbox(

            "✅ 이 값은 실제로 수행한 TLC 실험 결과입니다."
        )


        submitted = st.form_submit_button(

            "💾 Save Real Experiment",

            type="primary"
        )


    if submitted:

        try:

            if not confirm_real:

                raise ValueError(
                    "실제 실험 결과 확인란을 체크하세요."
                )


            mobile_phase = (
                normalize_mobile_phase([

                    (
                        rs1,
                        rr1
                    ),

                    (
                        rs2,
                        rr2
                    ),

                    (
                        rs3,
                        rr3
                    ),
                ])
            )


            row = create_experiment_row(

                compound_name=name,

                smiles=record_smiles,

                mobile_phase=mobile_phase,

                additive=record_additive,

                additive_percent=(
                    record_additive_percent
                ),

                plate=plate,

                concentration=(
                    record_concentration
                ),

                spotting_volume=(
                    record_spotting
                ),

                development_distance=(
                    development_distance
                ),

                chamber_saturated=(
                    chamber_saturated
                ),

                detection_method=detection,

                experimental_rf=(
                    experimental_rf
                ),

                tailing=tailing,

                spot_quality=spot_quality,

                notes=notes
            )


            db = (
                st.session_state[
                    "experiment_db"
                ]
            )


            db = pd.concat(

                [
                    db,
                    pd.DataFrame(
                        [
                            row
                        ]
                    )
                ],

                ignore_index=True,

                sort=False
            )


            st.session_state[
                "experiment_db"
            ] = db


            save_runtime_db()

            sync_engine_state()

            # An immediate, snappy confirmation the click registered --
            # the detailed st.success() below still explains WHAT was
            # saved and gives the Experiment ID, but that only appears
            # after the full rerun; this answers "did that work?" first.
            st.toast("실험 저장됨", icon="✓")

            st.success(
                "실제 TLC 실험이 저장되었습니다. (로컬 파일)"
            )


            st.write(
                "Experiment ID:",
                row[
                    "ExperimentID"
                ]
            )


            if st.session_state.get("db_access_token"):

                try:

                    cloud_body = build_cloud_tlc_body(
                        compound_name=name,
                        smiles=record_smiles,
                        mobile_phase=mobile_phase,
                        plate=plate,
                        development_distance_cm=development_distance,
                        chamber_saturated=chamber_saturated,
                        detection_method=detection,
                        experimental_rf=experimental_rf,
                        tailing_value=tailing,
                        spot_quality_value=spot_quality,
                        spotting_volume_ul=record_spotting,
                        notes_text=notes,
                    )

                    cloud_result = db_client.call_function(
                        "tlc",
                        st.session_state["db_access_token"],
                        {**cloud_body, "action": "create"}
                    )

                    st.success(
                        f"공유 데이터베이스에도 저장되었습니다. (실행 번호: {cloud_result['analysis']['analysis_id']})"
                    )

                except db_client.ApiError as cloud_error:

                    st.warning(
                        f"로컬 저장은 완료됐지만, 공유 데이터베이스 저장에는 실패했습니다: "
                        f"{db_client.friendly_error(cloud_error)}"
                    )

            else:

                st.caption(
                    "공유 데이터베이스에는 저장되지 않았습니다 (사이드바에서 로그인하면 함께 저장됩니다)."
                )


        except Exception as error:

            st.error(
                "실험 저장 실패"
            )

            st.exception(
                error
            )


# ============================================================
# TAB 6 — DATABASE / QC
# ============================================================

with tab_database:

    st.header(
        "🗂 Experiment Database / QC"
    )


    db = (
        st.session_state[
            "experiment_db"
        ]
    )


    a, b = st.columns(
        2
    )


    a.metric(
        "Experiment rows",
        len(
            db
        )
    )


    b.metric(

        "Unique compounds",

        (
            db[
                "SMILES"
            ].nunique()

            if (
                len(db) > 0
                and
                "SMILES"
                in db.columns
            )

            else 0
        )
    )


    st.subheader(
        "CSV Import"
    )


    uploaded_csv = st.file_uploader(

        "Real TLC experiment CSV",

        type=[
            "csv"
        ],

        key="experiment_csv_upload"
    )


    import_confirm = st.checkbox(

        "업로드한 CSV가 실제 TLC 실험 데이터임을 확인합니다.",

        key="import_confirm"
    )


    if st.button(
        "⬆️ Import CSV"
    ):

        try:

            if uploaded_csv is None:

                raise ValueError(
                    "CSV 파일을 선택하세요."
                )


            if not import_confirm:

                raise ValueError(
                    "실제 데이터 확인란을 체크하세요."
                )


            incoming = pd.read_csv(
                uploaded_csv
            )


            required = [

                "CompoundName",
                "SMILES",

                "Solvent1",
                "Solvent1Percent",

                "Experimental_Rf",

                "Tailing",
                "SpotQuality",
            ]


            missing = [

                column

                for column in required

                if column
                not in incoming.columns
            ]


            if missing:

                raise ValueError(

                    "필수 column이 없습니다: "
                    +
                    ", ".join(
                        missing
                    )
                )


            if (
                "DataOrigin"
                in incoming.columns
            ):

                invalid = (

                    incoming[
                        "DataOrigin"
                    ]

                    .astype(str)

                    .str.strip()

                    !=
                    "RealExperiment"
                )


                if invalid.any():

                    raise ValueError(
                        "DataOrigin은 RealExperiment여야 합니다."
                    )


            incoming[
                "DataOrigin"
            ] = "RealExperiment"


            # 필요한 column 보완
            for column in EXPERIMENT_COLUMNS:

                if column not in incoming.columns:

                    incoming[
                        column
                    ] = np.nan


            # ID가 없으면 생성
            for index in incoming.index:

                value = incoming.at[
                    index,
                    "ExperimentID"
                ]


                if (
                    pd.isna(
                        value
                    )
                    or
                    str(
                        value
                    ).strip()
                    in {
                        "",
                        "nan",
                    }
                ):

                    incoming.at[
                        index,
                        "ExperimentID"
                    ] = (

                        "IMPORT_"
                        +
                        uuid.uuid4().hex
                    )


            current = (

                st.session_state[
                    "experiment_db"
                ]
                .copy()
            )


            existing_ids = set(

                current[
                    "ExperimentID"
                ].astype(str)

                if (
                    len(current) > 0
                    and
                    "ExperimentID"
                    in current.columns
                )

                else []
            )


            new_rows = incoming[
                ~incoming[
                    "ExperimentID"
                ]
                .astype(str)
                .isin(
                    existing_ids
                )
            ].copy()


            merged = pd.concat(

                [
                    current,
                    new_rows
                ],

                ignore_index=True,

                sort=False
            )


            st.session_state[
                "experiment_db"
            ] = merged


            save_runtime_db()

            sync_engine_state()


            st.success(
                f"{len(new_rows)}개의 새 실험을 추가했습니다."
            )


        except Exception as error:

            st.error(
                "CSV Import 실패"
            )

            st.exception(
                error
            )


    st.divider()


    if st.button(
        "🔍 Run QC"
    ):

        try:

            qc_db = (
                run_current_qc()
            )


            if len(qc_db) == 0:

                st.info(
                    "QC할 실험이 없습니다."
                )

            else:

                pass_count = int(

                    (
                        qc_db[
                            "QCStatus"
                        ]
                        ==
                        "PASS"
                    ).sum()
                )


                review_count = int(

                    (
                        qc_db[
                            "QCStatus"
                        ]
                        ==
                        "REVIEW"
                    ).sum()
                )


                fail_count = int(

                    (
                        qc_db[
                            "QCStatus"
                        ]
                        ==
                        "FAIL"
                    ).sum()
                )


                q1, q2, q3 = (
                    st.columns(
                        3
                    )
                )


                q1.metric(
                    "PASS",
                    pass_count
                )


                q2.metric(
                    "REVIEW",
                    review_count
                )


                q3.metric(
                    "FAIL",
                    fail_count
                )


                st.success(
                    "QC 완료"
                )


        except Exception as error:

            st.error(
                "QC 실패"
            )

            st.exception(
                error
            )


    db = (
        st.session_state[
            "experiment_db"
        ]
    )


    st.subheader(
        "Current database"
    )


    st.dataframe(
        db,
        use_container_width=True,
        height=400
    )


    csv_bytes = (
        db.to_csv(
            index=False
        )
        .encode(
            "utf-8"
        )
    )


    st.download_button(

        "⬇️ Download Full Database",

        data=csv_bytes,

        file_name="LNP_TLC_REAL_DATABASE.csv",

        mime="text/csv"
    )


    if (
        len(db) > 0
        and
        "UseForRfTraining"
        in db.columns
    ):

        candidates = db[
            db[
                "UseForRfTraining"
            ] == True
        ].copy()


        st.download_button(

            "⬇️ Download Training Candidates",

            data=(
                candidates
                .to_csv(
                    index=False
                )
                .encode(
                    "utf-8"
                )
            ),

            file_name=(
                "LNP_TLC_TRAINING_CANDIDATES.csv"
            ),

            mime="text/csv"
        )


# ============================================================
# TAB 7 — MODEL TRAINING
# ============================================================

with tab_training:

    st.header(
        "🧠 Model Training"
    )


    st.warning(
        "모델이 ACTIVE가 되었다고 해서 "
        "LNP에 대해 검증 완료라는 뜻은 아닙니다. "
        "학습에 사용하지 않은 lipid로 별도 검증해야 합니다."
    )


    try:

        qc_preview = (
            run_current_qc()
        )

    except Exception:

        qc_preview = (
            st.session_state[
                "experiment_db"
            ]
        )


    total_rows = len(
        qc_preview
    )


    correction_eligible = (

        int(
            qc_preview[
                "UseForCorrectionTraining"
            ].fillna(
                False
            ).sum()
        )

        if (
            total_rows > 0
            and
            "UseForCorrectionTraining"
            in qc_preview.columns
        )

        else 0
    )


    tailing_eligible = (

        int(
            qc_preview[
                "UseForTailingTraining"
            ].fillna(
                False
            ).sum()
        )

        if (
            total_rows > 0
            and
            "UseForTailingTraining"
            in qc_preview.columns
        )

        else 0
    )


    m1, m2, m3 = st.columns(
        3
    )


    m1.metric(
        "Real experiments",
        total_rows
    )


    m2.metric(
        "Correction eligible",
        correction_eligible
    )


    m3.metric(
        "Tailing eligible",
        tailing_eligible
    )


    st.subheader(
        "1. Prepare ML Dataset"
    )


    if st.button(
        "🧬 Prepare Training Dataset"
    ):

        try:

            rf_data = (
                prepare_rf_dataset()
            )


            correction_data = (
                prepare_correction_dataset()
            )


            tailing_data = (
                prepare_tailing_dataset()
            )


            st.success(
                "Training dataset 생성 완료"
            )


            d1, d2, d3 = (
                st.columns(
                    3
                )
            )


            d1.metric(
                "Rf dataset",
                len(
                    rf_data
                )
            )


            d2.metric(
                "Correction dataset",
                len(
                    correction_data
                )
            )


            d3.metric(
                "Tailing dataset",
                len(
                    tailing_data
                )
            )


            if len(
                correction_data
            ) > 0:

                st.dataframe(

                    correction_data.head(
                        30
                    ),

                    use_container_width=True
                )


        except Exception as error:

            st.error(
                "Dataset 준비 실패"
            )

            st.exception(
                error
            )


    st.divider()


    st.subheader(
        "2. LNP Rf Correction Model"
    )


    st.caption(
        "현재 프로젝트 safety gate: "
        "최소 20 observations + 5 unique lipids"
    )


    correction_confirm = st.checkbox(

        "실제 데이터를 이용해 correction model을 학습합니다.",

        key="correction_train_confirm"
    )


    if st.button(
        "🧠 Train LNP Correction Model"
    ):

        try:

            if not correction_confirm:

                raise ValueError(
                    "학습 확인란을 체크하세요."
                )


            (
                model,
                data,
                n_rows,
                n_lipids,
                metrics,
                split_warning

            ) = train_correction_model_app()


            st.success(
                "LNP correction model 학습 완료"
            )


            st.write(
                f"Rows: **{n_rows}**"
            )


            st.write(
                f"Unique lipids: **{n_lipids}**"
            )


            if metrics:
                st.markdown("**Held-out 검증 (학습에 쓰지 않은 지질로 평가)**")
                m1, m2, m3 = st.columns(3)
                m1.metric("Held-out R²", f"{metrics['r2']:.4f}")
                m2.metric("Held-out RMSE", f"{metrics['rmse']:.4f}")
                m3.metric("Held-out MAE", f"{metrics['mae']:.4f}")
                st.caption(
                    f"검증에 쓴 지질 {metrics['n_test_lipids']}종, "
                    f"{metrics['n_test_rows']}행 — 학습 데이터에는 포함되지 않았습니다."
                )
                st.info(
                    "배포된 모델은 이 지표를 잰 뒤 전체 데이터로 다시 학습한 것입니다 "
                    "(검증 세트만큼 데이터를 버리고 배포하지 않기 위함)."
                )
            else:
                st.warning(
                    "ACTIVE BUT UNVALIDATED — "
                    f"{split_warning or 'unseen lipid validation이 아직 필요합니다.'}"
                )


        except Exception as error:

            st.error(
                "Correction model을 학습하지 않았습니다."
            )

            st.exception(
                error
            )


    st.divider()


    st.subheader(
        "3. Tailing Model"
    )


    st.caption(
        "현재 프로젝트 safety gate: "
        "최소 30 observations + 5 unique lipids + "
        "각 class 최소 5 observations"
    )


    tailing_confirm = st.checkbox(

        "실제 데이터를 이용해 tailing model을 학습합니다.",

        key="tailing_train_confirm"
    )


    if st.button(
        "🧠 Train Tailing Model"
    ):

        try:

            if not tailing_confirm:

                raise ValueError(
                    "학습 확인란을 체크하세요."
                )


            (
                model,
                data,
                n_rows,
                n_lipids,
                class0,
                class1,
                metrics,
                split_warning

            ) = train_tailing_model_app()


            st.success(
                "Tailing model 학습 완료"
            )


            st.write(
                f"Rows: **{n_rows}**"
            )


            st.write(
                f"Unique lipids: **{n_lipids}**"
            )


            st.write(
                f"No/Mild: **{class0}**"
            )


            st.write(
                f"Moderate/Severe: **{class1}**"
            )


            if metrics:
                st.markdown("**Held-out 검증 (학습에 쓰지 않은 지질로 평가)**")
                m1, m2, m3 = st.columns(3)
                m1.metric("Held-out Accuracy", f"{metrics['accuracy']:.3f}")
                m2.metric("Held-out Balanced Accuracy", f"{metrics['balanced_accuracy']:.3f}")
                m3.metric("Held-out F1", f"{metrics['f1']:.3f}")
                st.caption(
                    f"검증에 쓴 지질 {metrics['n_test_lipids']}종, "
                    f"{metrics['n_test_rows']}행 — 학습 데이터에는 포함되지 않았습니다."
                )
                st.info(
                    "배포된 모델은 이 지표를 잰 뒤 전체 데이터로 다시 학습한 것입니다 "
                    "(검증 세트만큼 데이터를 버리고 배포하지 않기 위함)."
                )
            else:
                st.warning(
                    "ACTIVE BUT UNVALIDATED — "
                    f"{split_warning or 'unseen lipid validation이 아직 필요합니다.'}"
                )


        except Exception as error:

            st.error(
                "Tailing model을 학습하지 않았습니다."
            )

            st.exception(
                error
            )


# ============================================================
# TAB 8 — MODEL STATUS
# ============================================================

with tab_status:

    st.header(
        "📊 Model Status"
    )


    st.subheader(
        "Global TLC model"
    )


    st.success(
        "ACTIVE"
    )


    g1, g2, g3 = st.columns(
        3
    )


    g1.metric(
        "Frozen test MAE",
        "0.0644"
    )


    g2.metric(
        "Frozen test RMSE",
        "0.1006"
    )


    g3.metric(
        "Frozen test R²",
        "0.9054"
    )


    st.caption(
        "이 수치는 public small-molecule TLC test set에서의 "
        "frozen performance입니다."
    )


    st.divider()


    correction_active = (

        st.session_state[
            "lnp_correction_model"
        ]

        is not None
    )


    tailing_active = (

        st.session_state[
            "tailing_model"
        ]

        is not None
    )


    s1, s2 = st.columns(
        2
    )


    with s1:

        st.subheader(
            "LNP correction"
        )


        if correction_active:

            st.warning(
                "🧠 ACTIVE — validation still required"
            )

        else:

            st.info(
                "⏳ NOT ACTIVE"
            )


    with s2:

        st.subheader(
            "Tailing model"
        )


        if tailing_active:

            st.warning(
                "🧠 ACTIVE — validation still required"
            )

        else:

            st.info(
                "⏳ NOT ACTIVE"
            )


    st.divider()


    db = (
        st.session_state[
            "experiment_db"
        ]
    )


    st.subheader(
        "Real experiment database"
    )


    st.metric(
        "Rows",
        len(
            db
        )
    )


    if (
        len(db) > 0
        and
        "QCStatus"
        in db.columns
    ):

        counts = (
            db[
                "QCStatus"
            ]
            .value_counts()
        )


        c1, c2, c3 = (
            st.columns(
                3
            )
        )


        c1.metric(
            "PASS",
            int(
                counts.get(
                    "PASS",
                    0
                )
            )
        )


        c2.metric(
            "REVIEW",
            int(
                counts.get(
                    "REVIEW",
                    0
                )
            )
        )


        c3.metric(
            "FAIL",
            int(
                counts.get(
                    "FAIL",
                    0
                )
            )
        )


    st.divider()


    st.subheader(
        "Interpretation"
    )


    st.markdown(
        """
- **IN-DOMAIN / supported solvent region**: 상대적으로 신뢰도가 높습니다.
- **HIGH OOD RISK**: 현재 public TLC 데이터와 구조적으로 많이 다릅니다.
- Ionizable lipid의 HIGH OOD 예측값은 **실험 탐색용**으로 취급합니다.
- 실제 LNP 데이터를 누적하면 correction/tailing 모델을 추가로 학습할 수 있습니다.
- 새로 학습한 모델은 반드시 **학습에 사용하지 않은 lipid**로 검증해야 합니다.
        """
    )
