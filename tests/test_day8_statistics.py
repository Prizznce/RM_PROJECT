import numpy as np
import pytest
from scipy import stats

def compute_odds_ratio(b, c):
    """Woolf method for Odds Ratio of discordant pairs in McNemar test."""
    if c == 0 or b == 0:
        return float('inf'), (0.0, float('inf'))
    or_val = b / c
    se = np.sqrt(1.0 / b + 1.0 / c)
    ci_low = np.exp(np.log(or_val) - 1.96 * se)
    ci_high = np.exp(np.log(or_val) + 1.96 * se)
    return or_val, (ci_low, ci_high)

def apply_holm_bonferroni(p_values):
    """Step-down Holm-Bonferroni correction."""
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    adj_p = np.zeros(m)
    
    for rank, idx in enumerate(sorted_indices):
        multiplier = m - rank
        adj_p[idx] = min(1.0, p_values[idx] * multiplier)
        
    # Enforce monotonicity
    running_max = 0.0
    for idx in sorted_indices:
        running_max = max(running_max, adj_p[idx])
        adj_p[idx] = min(1.0, running_max)
        
    return adj_p

def moving_block_bootstrap_sample(y_a, y_b, y_true, block_len=24, rng=None):
    """Samples moving blocks of length L to compute delta accuracy."""
    if rng is None:
        rng = np.random.RandomState(42)
    n = len(y_true)
    num_blocks = int(np.ceil(n / block_len))
    max_start = n - block_len + 1
    start_indices = rng.randint(0, max_start, size=num_blocks)
    
    boot_indices = []
    for s in start_indices:
        boot_indices.extend(range(s, s + block_len))
    boot_indices = np.array(boot_indices[:n])
    
    acc_a = np.mean(y_a[boot_indices] == y_true[boot_indices])
    acc_b = np.mean(y_b[boot_indices] == y_true[boot_indices])
    return acc_a - acc_b

def check_pareto_domination(p1_acc, p1_lat, p2_acc, p2_lat):
    """Returns True if p1 dominates p2 (higher acc, lower lat)."""
    return (p1_acc >= p2_acc and p1_lat <= p2_lat) and (p1_acc > p2_acc or p1_lat < p2_lat)

# ----------------- TESTS -----------------

def test_odds_ratio_and_ci():
    b = 460  # CGT-v3 correct, HT wrong
    c = 286  # HT correct, CGT-v3 wrong
    or_val, (ci_low, ci_high) = compute_odds_ratio(b, c)
    
    expected_or = 460 / 286
    assert pytest.approx(or_val, rel=1e-3) == expected_or
    assert ci_low < or_val < ci_high
    assert 1.35 < ci_low < 1.45
    assert 1.80 < ci_high < 1.90

def test_holm_bonferroni_properties():
    # 5 raw p-values
    raw_p = [0.001, 0.012, 0.024, 0.045, 0.080]
    adj_p = apply_holm_bonferroni(raw_p)
    
    # Smallest p (0.001) multiplied by 5 -> 0.005
    assert pytest.approx(adj_p[0], rel=1e-3) == 0.005
    # Second smallest (0.012) multiplied by 4 -> 0.048
    assert pytest.approx(adj_p[1], rel=1e-3) == 0.048
    # Monotonicity check: adj_p should be monotonically non-decreasing
    for i in range(len(adj_p) - 1):
        assert adj_p[i] <= adj_p[i + 1]
    assert np.all(adj_p <= 1.0)

def test_moving_block_bootstrap_invariants():
    rng = np.random.RandomState(42)
    n = 1000
    y_true = rng.randint(0, 2, size=n)
    # Model A has 90% accuracy, Model B has 80% accuracy
    y_a = y_true.copy()
    flip_a = rng.choice(n, size=100, replace=False)
    y_a[flip_a] = 1 - y_a[flip_a]
    
    y_b = y_true.copy()
    flip_b = rng.choice(n, size=200, replace=False)
    y_b[flip_b] = 1 - y_b[flip_b]
    
    delta = moving_block_bootstrap_sample(y_a, y_b, y_true, block_len=24, rng=rng)
    # Delta should be roughly +0.10 (+- 0.05)
    assert 0.05 < delta < 0.15

def test_friedman_average_rank_sum():
    # In any Friedman ranking of k models, the sum of average ranks must equal k*(k+1)/2
    k = 15
    N = 24
    rng = np.random.RandomState(42)
    fake_matrix = rng.rand(N, k)
    
    ranks = np.zeros_like(fake_matrix)
    for i in range(N):
        ranks[i, :] = stats.rankdata(-fake_matrix[i, :])
    avg_ranks = np.mean(ranks, axis=0)
    
    expected_sum = k * (k + 1) / 2.0
    assert pytest.approx(np.sum(avg_ranks), rel=1e-5) == expected_sum

def test_pareto_dominance_logic():
    # CGT-v3: 85.78% acc, 888 us
    # Always-HT: 85.07% acc, 98 us
    # A3: 89.30% acc, 1250 us
    # Dominated point: 84.00% acc, 1500 us
    
    # CGT-v3 does NOT dominate Always-HT (CGT-v3 has higher acc, but HT has lower latency)
    assert not check_pareto_domination(85.78, 888, 85.07, 98)
    assert not check_pareto_domination(85.07, 98, 85.78, 888)
    
    # Both CGT-v3 and Always-HT dominate the inferior point (84.00%, 1500 us)
    assert check_pareto_domination(85.78, 888, 84.00, 1500)
    assert check_pareto_domination(85.07, 98, 84.00, 1500)

