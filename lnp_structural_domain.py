"""LNP ionizable-lipid structural applicability domain.

Answers exactly one question: how structurally familiar is a query SMILES
relative to a known public LNP ionizable-lipid library (LNP Atlas or LNPDB)?
Uses Morgan fingerprints + Tanimoto similarity -- no neural network, no
training step. Full provenance for both reference datasets (source, license,
verification method, row counts) is in data/reference/lnp_atlas_manifest.json
and data/reference/lnpdb_manifest.json, and the full evaluation that produced
this module is in the platform repository's docs/16_LNP_DATASET_STRATEGY.md
(this app is deployed standalone, so that document does not travel with it --
the manifests are the portable record of where this data came from).

This module is intentionally separate from, and must never be merged into,
this app's own TLC Rf / tailing predictions. A lipid can look very familiar
here (high similarity to a well-studied scaffold like C12-200 or MC3) while
having zero recorded TLC data -- that is a statement about known LNP
chemistry, not about how well this app's own TLC model predicts its Rf.
Callers must display this as its own clearly-labeled section, never combine
its similarity score with `MoleculeDomain`/`StructuralSimilarity` (which is
the *general small-molecule TLC model's* own, unrelated domain check) into a
single number.
"""
import bisect
import csv
import random
from dataclasses import dataclass, field
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

RDLogger.DisableLog("rdApp.*")

_DATA_DIR = Path(__file__).resolve().parent / "data" / "reference"

SOURCES = {
    "lnp_atlas": {
        "csv_path": _DATA_DIR / "lnp_atlas_ionizable_lipids.csv",
        "display_name": "LNP Atlas",
    },
    "lnpdb": {
        "csv_path": _DATA_DIR / "lnpdb_ionizable_lipids.csv",
        "display_name": "LNPDB",
    },
}

_CACHE = {}
_THRESHOLD_CACHE = {}


def clear_cache():
    _CACHE.clear()
    _THRESHOLD_CACHE.clear()


def _compute_fingerprint(mol, radius=2, n_bits=2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def _load_reference(source_key):
    if source_key not in SOURCES:
        raise ValueError(f"unknown source_key {source_key!r}; expected one of {list(SOURCES)}")

    if source_key in _CACHE:
        return _CACHE[source_key]

    csv_path = SOURCES[source_key]["csv_path"]
    records = []
    fingerprints = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("is_first_occurrence_of_structure") != "True":
                continue
            mol = Chem.MolFromSmiles(row["canonical_smiles"])
            if mol is None:
                continue
            records.append(row)
            fingerprints.append(_compute_fingerprint(mol))

    _CACHE[source_key] = (records, fingerprints)
    return _CACHE[source_key]


def _internal_nearest_neighbor_distribution(fingerprints, sample_size=300, seed=20260816):
    """How similar is a 'typical' structure already in this library to its own
    nearest neighbor within the same library? Used as the data-derived basis for
    the in-domain/borderline/novel thresholds below, instead of a fixed constant
    -- an arbitrary cutoff was explicitly ruled out when this module was designed
    (see docs/16_LNP_DATASET_STRATEGY.md in the main platform repository).

    sample_size=300 here (vs. 1500 in the platform repository's own copy of this
    module) is a deliberate, smaller choice: this copy runs inside a live
    Streamlit rerun on first use, and the ~13k-structure LNPDB library made the
    1500-sample version cost several extra seconds on top of this app's own
    already-heavy model-loading step. 300 samples is still enough to place a
    median/10th-percentile estimate reasonably, and the result is cached after
    the first call regardless (see _get_thresholds), so this only affects the
    very first assessment made against a given library in a running process.
    """
    n = len(fingerprints)
    if n < 2:
        return []
    rng = random.Random(seed)
    idx_sample = rng.sample(range(n), min(sample_size, n))
    scores = []
    for i in idx_sample:
        sims = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints)
        sims[i] = -1.0
        scores.append(max(sims))
    return sorted(scores)


def _percentile(sorted_values, value):
    if not sorted_values:
        return None
    pos = bisect.bisect_left(sorted_values, value)
    return round(100.0 * pos / len(sorted_values), 1)


def _get_thresholds(source_key, fingerprints):
    """Cached wrapper around _internal_nearest_neighbor_distribution: this
    distribution depends only on the reference library, never on the query, so
    it must be computed once per library and reused -- not recomputed on every
    single assess_structural_domain() call, which on a ~13k-structure library
    (LNPDB) is expensive enough to matter inside a Streamlit rerun loop.
    """
    if source_key not in _THRESHOLD_CACHE:
        internal_dist = _internal_nearest_neighbor_distribution(fingerprints)
        if internal_dist:
            median = internal_dist[len(internal_dist) // 2]
            p10 = internal_dist[max(0, int(0.10 * len(internal_dist)))]
        else:
            median = p10 = None
        _THRESHOLD_CACHE[source_key] = (internal_dist, median, p10)
    return _THRESHOLD_CACHE[source_key]


@dataclass
class DomainAssessment:
    query_smiles: str
    source_key: str
    source_display_name: str
    valid: bool
    error: str = None
    top_similarity: float = None
    mean_top_k_similarity: float = None
    nearest_neighbors: list = field(default_factory=list)
    percentile_vs_reference_internal_distribution: float = None
    domain_category: str = None
    reference_size: int = None
    threshold_basis: str = None


def assess_structural_domain(smiles, source_key="lnp_atlas", top_k=5):
    """Assess how structurally familiar `smiles` is relative to the named public
    LNP reference library. Never raises on a bad SMILES or unknown source -- both
    return valid=False with an error message, since this sits behind a UI field.
    """
    display_name = SOURCES.get(source_key, {}).get("display_name", source_key)

    if source_key not in SOURCES:
        return DomainAssessment(
            query_smiles=smiles, source_key=source_key, source_display_name=display_name,
            valid=False, error=f"unknown source_key {source_key!r}",
        )

    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return DomainAssessment(
            query_smiles=smiles, source_key=source_key, source_display_name=display_name,
            valid=False, error="RDKit could not parse this SMILES string",
        )

    records, fingerprints = _load_reference(source_key)
    if not fingerprints:
        return DomainAssessment(
            query_smiles=smiles, source_key=source_key, source_display_name=display_name,
            valid=False, error=f"reference set '{source_key}' has no usable structures",
        )

    query_fp = _compute_fingerprint(mol)
    sims = DataStructs.BulkTanimotoSimilarity(query_fp, fingerprints)
    ranked = sorted(zip(sims, records), key=lambda x: x[0], reverse=True)
    top = ranked[:top_k]
    top1 = top[0][0]

    internal_dist, median, p10 = _get_thresholds(source_key, fingerprints)
    pct = _percentile(internal_dist, top1)

    if median is not None and top1 >= median:
        category = "in_domain"
    elif p10 is not None and top1 >= p10:
        category = "borderline"
    else:
        category = "novel"

    threshold_basis = (
        f"in_domain: top-1 similarity >= library median self-similarity ({median:.3f}); "
        f"borderline: >= library 10th-percentile self-similarity ({p10:.3f})"
        if median is not None else "insufficient reference data to derive a threshold"
    )

    return DomainAssessment(
        query_smiles=smiles,
        source_key=source_key,
        source_display_name=display_name,
        valid=True,
        top_similarity=round(top1, 4),
        mean_top_k_similarity=round(sum(s for s, _ in top) / len(top), 4),
        nearest_neighbors=[
            {
                "similarity": round(s, 4),
                "source_name": r.get("source_name"),
                "canonical_smiles": r.get("canonical_smiles"),
            }
            for s, r in top
        ],
        percentile_vs_reference_internal_distribution=pct,
        domain_category=category,
        reference_size=len(fingerprints),
        threshold_basis=threshold_basis,
    )
