import numpy as np
import logging
from zeromodel.organization import (
    MemoryOrganizationStrategy,
    SqlOrganizationStrategy,
    ZeroModelOrganizationStrategy,
)
from zeromodel.organization.duckdb_adapter import DuckDBAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_strategy_comparison(sql_task, mem_task, score_matrix, metric_names):
    sql_strategy = SqlOrganizationStrategy()
    mem_strategy = MemoryOrganizationStrategy()

    # FIX: enforce correct param order for adapter
    sql_strategy.adapter.execute(score_matrix, metric_names, sql_task)

    mem_results = mem_strategy.organize(score_matrix, metric_names, mem_task)
    sql_results = sql_strategy.organize(score_matrix, metric_names, sql_task)

    # compare
    assert sql_results["doc_order"] == mem_results["doc_order"]
    assert sql_results["metric_order"] == mem_results["metric_order"]
    return sql_results


def assert_consistency(results, reference="memory"):
    """Assert that all strategies match the reference ordering."""
    ref_order = results[reference]["doc_order"]
    ref_matrix = results[reference]["sorted_matrix"]

    for name, res in results.items():
        assert np.array_equal(res["doc_order"], ref_order), \
            f"{name} strategy doc_order differs from {reference}"
        assert np.allclose(res["sorted_matrix"], ref_matrix), \
            f"{name} strategy sorted_matrix differs from {reference}"
        assert "ordering" in res["analysis"]


def test_organization_strategies_consistency():
    """Verify all organization strategies produce consistent results."""
    np.random.seed(42)
    n_docs, n_metrics = 100, 4
    score_matrix = np.random.rand(n_docs, n_metrics)
    metric_names = [f"metric{i}" for i in range(n_metrics)]

    sql_task = "SELECT * FROM virtual_index ORDER BY metric0 DESC"
    mem_task = "metric0 DESC"

    results = run_strategy_comparison(sql_task, mem_task, score_matrix, metric_names)
    assert_consistency(results)


def test_organization_strategies_different_tasks():
    """Test multiple metrics and directions."""
    np.random.seed(42)
    score_matrix = np.random.rand(50, 3)
    metric_names = ["accuracy", "uncertainty", "size"]

    tasks = [
        ("SELECT * FROM virtual_index ORDER BY accuracy DESC", "accuracy DESC"),
        ("SELECT * FROM virtual_index ORDER BY uncertainty ASC", "uncertainty ASC"),
        ("SELECT * FROM virtual_index ORDER BY size DESC", "size DESC"),
    ]

    for sql_task, mem_task in tasks:
        results = run_strategy_comparison(sql_task, mem_task, score_matrix, metric_names)
        assert_consistency(results)


def test_edge_case_single_row():
    """Edge case: only one row."""
    score_matrix = np.array([[0.9]])
    metric_names = ["metric1"]

    sql_task = "SELECT * FROM virtual_index ORDER BY metric1 DESC"
    mem_task = "metric1 DESC"

    results = run_strategy_comparison(sql_task, mem_task, score_matrix, metric_names)
    assert_consistency(results)
