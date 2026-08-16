from lnp_structural_domain import assess_structural_domain, clear_cache, _load_reference


def setup_function(_):
    clear_cache()


def test_known_lnp_atlas_lipid_is_in_domain_against_its_own_library():
    # Pull DLin-MC3-DMA's own canonical SMILES out of the bundled reference set
    # itself, rather than hand-typing a remembered structure -- feed it back in
    # as a query, it must self-match at similarity 1.0.
    records, _ = _load_reference("lnp_atlas")
    mc3_row = next(r for r in records if r["source_name"] == "DLin-MC3-DMA")
    result = assess_structural_domain(mc3_row["canonical_smiles"], source_key="lnp_atlas")
    assert result.valid is True
    assert result.top_similarity == 1.0
    assert result.domain_category == "in_domain"


def test_unrelated_small_molecule_is_novel():
    aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    result = assess_structural_domain(aspirin_smiles, source_key="lnp_atlas")
    assert result.valid is True
    assert result.domain_category == "novel"


def test_invalid_smiles_reports_a_clean_error_not_a_crash():
    result = assess_structural_domain("not a smiles string", source_key="lnpdb")
    assert result.valid is False
    assert result.error is not None


def test_unknown_source_key_reports_a_clean_error():
    result = assess_structural_domain("CCO", source_key="nonexistent")
    assert result.valid is False
    assert "unknown source_key" in result.error


def test_threshold_is_derived_from_data_not_hardcoded():
    result = assess_structural_domain("CCO", source_key="lnpdb")
    assert result.valid is True
    assert "library median self-similarity" in result.threshold_basis


def test_reference_sizes_match_the_platform_repository_manifest():
    # Regression pin: docs/16_LNP_DATASET_STRATEGY.md in the main platform
    # repository records 259 unique LNP Atlas structures and 12,837 unique
    # LNPDB structures. If this ever drifts, the bundled CSV here and the
    # platform repository's copy have gone out of sync.
    atlas_records, _ = _load_reference("lnp_atlas")
    lnpdb_records, _ = _load_reference("lnpdb")
    assert len(atlas_records) == 259
    assert len(lnpdb_records) == 12837
