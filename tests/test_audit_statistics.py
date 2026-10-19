"""
Tests for independent statistical audit utilities in scripts/audit_monte_carlo_statistics.py.
"""

import math
import pytest
from scripts.audit_monte_carlo_statistics import (
    wilson_score_interval,
    student_t_pvalue,
    wilcoxon_signed_rank,
    kaplan_meier_rmst,
)


def test_wilson_score_interval_bounds():
    # 0 successes
    low, high = wilson_score_interval(0, 30)
    assert low == 0.0
    assert 0.0 < high < 0.15

    # 30 successes out of 30
    low, high = wilson_score_interval(30, 30)
    assert 0.85 < low < 1.0
    assert high == 1.0

    # 15 successes out of 30
    low, high = wilson_score_interval(15, 30)
    assert 0.30 < low < 0.40
    assert 0.60 < high < 0.70
    assert abs((low + high) / 2.0 - 0.5) < 0.01


def test_student_t_pvalue_precision():
    # df=29 critical values
    p_crit_05 = student_t_pvalue(2.045229642, df=29)
    assert abs(p_crit_05 - 0.05) < 1e-6

    p_crit_01 = student_t_pvalue(2.756385904, df=29)
    assert abs(p_crit_01 - 0.01) < 1e-6

    # Extreme value does not underflow to 0 or crash
    p_extreme = student_t_pvalue(20.0, df=29)
    assert 0.0 < p_extreme < 1e-15


def test_wilcoxon_signed_rank_ties_and_zeros():
    # Identical pairs (all zeros)
    res_zero = wilcoxon_signed_rank([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert res_zero["n_nz"] == 0
    assert res_zero["p_val_cc"] == 1.0

    # Strictly positive differences
    x = [10.0 + i for i in range(10)]
    y = [5.0 + i for i in range(10)]
    res_pos = wilcoxon_signed_rank(x, y)
    assert res_pos["w_stat"] == 0.0
    assert res_pos["p_val_cc"] < 0.01


def test_kaplan_meier_rmst_properties():
    # All events occur at t=10s
    durations = [10.0] * 10
    events = [True] * 10
    km = kaplan_meier_rmst(durations, events, horizon=100.0)
    assert km["discovery_rate"] == 1.0
    assert km["median_ttd"] == 10.0
    assert abs(km["rmst"] - 10.0) < 1e-5

    # All censored at horizon=100s
    durations_cens = [100.0] * 10
    events_cens = [False] * 10
    km_cens = kaplan_meier_rmst(durations_cens, events_cens, horizon=100.0)
    assert km_cens["discovery_rate"] == 0.0
    assert km_cens["median_ttd"] is None
    assert abs(km_cens["rmst"] - 100.0) < 1e-5
