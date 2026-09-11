"""Chapter 11 fast loop: the parts of the loader that need no registry.

Model loading is exercised by make ch11-validate, which is allowed a network
dependency. This suite stays offline, like the Chapter 9 and 10 suites.
"""

import numpy as np

from src.serving import artifacts as A


def test_the_threshold_is_read_from_chapter_9_at_full_precision():
    threshold, rule = A._load_threshold()
    assert rule == "example-dependent, global"
    assert threshold == 0.08704082880739572
    # trap 13: Chapter 10 lost one transaction to six decimals
    assert round(threshold, 6) != threshold


def test_the_calibrator_is_chapter_9s_object_and_is_strictly_increasing():
    cal, factor = A._load_calibrator()
    assert cal.kind == "platt"
    assert factor == 1.0
    p = np.linspace(1e-6, 1 - 1e-6, 10_000)
    assert np.all(np.diff(cal.transform(p)) > 0)


def test_the_chapter_10_maps_agree_with_the_contract():
    family_of, group_of, order = A._load_maps()
    contract = A._load_contract()
    assert order == contract
    assert set(family_of) == set(contract)
    assert len(group_of) == 143
    assert set(group_of) <= set(contract)
    assert len(set(group_of.values())) == 13
    assert len(set(family_of.values())) == 12


def test_anonymised_features_render_as_groups_and_named_ones_do_not():
    _, group_of, _ = A._load_maps()
    assert A.render_feature("v70", group_of) == "V-group 05"
    assert A.render_feature("transactionamt", group_of) == "transactionamt"