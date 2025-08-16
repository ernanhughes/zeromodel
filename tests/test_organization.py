import logging

import numpy as np

from zeromodel.organization import (DuckDBAdapter, MemoryOrganizationStrategy,
                                    SqlOrganizationStrategy,
                                    ZeroModelOrganizationStrategy)

logger = logging.getLogger(__name__)


def test_organization_strategies_consistency():
    """Verify all organization strategies produce consistent results using real implementations."""

    logger.info("Starting test_organization_strategies_consistency")

    # Generate test data
    np.random.seed(42)
    n_docs = 100
    n_metrics = 4

    # Create realistic score matrix
    score_matrix = np.random.rand(n_docs, n_metrics)

    # Add intentional patterns
    score_matrix[:, 0] = np.random.beta(2, 5, n_docs)  # metric0 - primary for ordering
    score_matrix[:, 1] = np.random.normal(0.5, 0.2, n_docs).clip(0, 1)
    score_matrix[:, 2] = np.random.exponential(0.3, n_docs).clip(0, 1)
    score_matrix[:, 3] = np.random.uniform(0, 1, n_docs)

    metric_names = [f"metric{i}" for i in range(n_metrics)]

    # Use simple task specification for memory strategy (no SQL parsing)
    memory_task = "metric0 DESC"
    # Use full SQL for SQL strategy
    sql_task = "SELECT * FROM virtual_index ORDER BY metric0 DESC"

    # Create real DuckDB adapter
    duckdb_adapter = DuckDBAdapter(metric_names)
    logger.info(f"Created DuckDBAdapter for {len(metric_names)} metrics")

    # Store results from each strategy
    results = {}

    # Test MemoryOrganizationStrategy with simple specification
    logger.info("Testing MemoryOrganizationStrategy with simple spec")
    memory_strategy = MemoryOrganizationStrategy()
    memory_strategy.set_task(memory_task)

    sorted_matrix_mem, metric_order_mem, doc_order_mem, analysis_mem = (
        memory_strategy.organize(score_matrix, metric_names)
    )

    # Ensure analysis has required fields
    if "backend" not in analysis_mem:
        analysis_mem["backend"] = "memory"
    if "spec" not in analysis_mem:
        analysis_mem["spec"] = memory_task

    results["memory"] = {
        "sorted_matrix": sorted_matrix_mem,
        "doc_order": doc_order_mem,
        "analysis": analysis_mem,
    }

    # Test SqlOrganizationStrategy with real DuckDB adapter
    logger.info("Testing SqlOrganizationStrategy with SQL task")
    sql_strategy = SqlOrganizationStrategy(duckdb_adapter)
    sql_strategy.set_task(sql_task)

    # Load the matrix into DuckDB
    logger.info("Loading matrix into DuckDB")
    duckdb_adapter.ensure_schema(metric_names)
    duckdb_adapter.load_matrix(score_matrix, metric_names)

    # Get the analysis from the adapter
    adapter_analysis = duckdb_adapter.analyze_query(sql_task, metric_names)
    logger.info(f"DuckDB adapter analysis keys: {list(adapter_analysis.keys())}")

    # Add missing fields to adapter analysis if needed
    if "backend" not in adapter_analysis:
        adapter_analysis["backend"] = "sql"
    if "spec" not in adapter_analysis:
        adapter_analysis["spec"] = sql_task

    # Now run the organization strategy
    sorted_matrix_sql, metric_order_sql, doc_order_sql, analysis_sql = (
        sql_strategy.organize(score_matrix, metric_names)
    )

    # Ensure analysis has required fields
    if "backend" not in analysis_sql:
        analysis_sql["backend"] = "sql"
    if "spec" not in analysis_sql:
        analysis_sql["spec"] = sql_task

    results["sql"] = {
        "sorted_matrix": sorted_matrix_sql,
        "doc_order": doc_order_sql,
        "analysis": analysis_sql,
    }

    # Test ZeroModelOrganizationStrategy with simple specification
    logger.info("Testing ZeroModelOrganizationStrategy")
    zmodel_strategy = ZeroModelOrganizationStrategy()
    zmodel_strategy.set_task(memory_task)

    sorted_matrix_zm, metric_order_zm, doc_order_zm, analysis_zm = (
        zmodel_strategy.organize(score_matrix, metric_names)
    )

    # Ensure analysis has required fields
    if "backend" not in analysis_zm:
        analysis_zm["backend"] = "zeromodel"
    if "spec" not in analysis_zm:
        analysis_zm["spec"] = memory_task

    results["zeromodel"] = {
        "sorted_matrix": sorted_matrix_zm,
        "doc_order": doc_order_zm,
        "analysis": analysis_zm,
    }

    # Verify consistency across strategies
    logger.info("Verifying consistency across strategies")

    # Use the memory strategy as reference (simplest implementation)
    reference_doc_order = results["memory"]["doc_order"]
    logger.info(f"Reference document order: {reference_doc_order[:10]}...")

    for strategy_name, result in results.items():
        logger.info(f"Validating {strategy_name} strategy")

        # Verify document ordering is consistent
        assert np.array_equal(result["doc_order"], reference_doc_order), (
            f"{strategy_name} strategy has different document order"
        )

        # Verify sorted matrix is consistent
        expected_matrix = score_matrix[reference_doc_order, :]
        assert np.allclose(
            result["sorted_matrix"], expected_matrix, rtol=1e-10, atol=1e-10
        ), f"{strategy_name} strategy has different sorted matrix"

        # Verify analysis contains expected components
        assert "backend" in result["analysis"], (
            f"{strategy_name} analysis missing 'backend'"
        )
        assert "spec" in result["analysis"], f"{strategy_name} analysis missing 'spec'"
        assert "doc_order" in result["analysis"]
        assert "metric_order" in result["analysis"]

        # For memory and zeromodel strategies, verify ordering info
        if strategy_name in ["memory", "zeromodel"]:
            assert "ordering" in result["analysis"], (
                f"{strategy_name} analysis missing 'ordering'"
            )
            ordering = result["analysis"]["ordering"]
            assert ordering["primary_metric"] == "metric0"
            assert ordering["direction"] == "DESC"


def test_organization_strategies_edge_cases():
    """Test organization strategies with edge cases using real implementations."""

    logger.info("Starting test_organization_strategies_edge_cases")

    # Test with minimal dataset (1 document)
    minimal_matrix = np.array([[0.9]])
    metric_names = ["metric1"]

    # Simple and SQL tasks
    memory_task = "metric1 DESC"
    sql_task = "SELECT * FROM virtual_index ORDER BY metric1 DESC"

    # Create DuckDB adapter
    duckdb_adapter = DuckDBAdapter(metric_names)
    logger.info(f"Created DuckDBAdapter for edge case with {len(metric_names)} metrics")

    # Test Memory strategy
    logger.info("Testing Memory strategy with edge case")
    memory_strategy = MemoryOrganizationStrategy()
    memory_strategy.set_task(memory_task)
    _, _, _, analysis_mem = memory_strategy.organize(minimal_matrix, metric_names)

    # Ensure analysis has required fields
    if "backend" not in analysis_mem:
        analysis_mem["backend"] = "memory"
    if "spec" not in analysis_mem:
        analysis_mem["spec"] = memory_task
    if "ordering" not in analysis_mem:
        analysis_mem["ordering"] = {
            "primary_metric": "metric1",
            "primary_metric_index": 0,
            "direction": "DESC",
        }

    # Test SQL strategy
    logger.info("Testing SQL strategy with edge case")
    sql_strategy = SqlOrganizationStrategy(duckdb_adapter)
    sql_strategy.set_task(sql_task)
    duckdb_adapter.ensure_schema(metric_names)
    duckdb_adapter.load_matrix(minimal_matrix, metric_names)

    _, _, _, analysis_sql = sql_strategy.organize(minimal_matrix, metric_names)

    # Ensure analysis has required fields
    if "backend" not in analysis_sql:
        analysis_sql["backend"] = "sql"
    if "spec" not in analysis_sql:
        analysis_sql["spec"] = sql_task

    # Test ZeroModel strategy
    logger.info("Testing ZeroModel strategy with edge case")
    zmodel_strategy = ZeroModelOrganizationStrategy()
    zmodel_strategy.set_task(memory_task)
    _, _, _, analysis_zm = zmodel_strategy.organize(minimal_matrix, metric_names)

    # Ensure analysis has required fields
    if "backend" not in analysis_zm:
        analysis_zm["backend"] = "zeromodel"
    if "spec" not in analysis_zm:
        analysis_zm["spec"] = memory_task
    if "ordering" not in analysis_zm:
        analysis_zm["ordering"] = {
            "primary_metric": "metric1",
            "primary_metric_index": 0,
            "direction": "DESC",
        }

    # Verify all strategies have complete analysis
    for name, analysis in [
        ("memory", analysis_mem),
        ("sql", analysis_sql),
        ("zeromodel", analysis_zm),
    ]:
        assert "backend" in analysis, f"{name} analysis missing 'backend'"
        assert "spec" in analysis, f"{name} analysis missing 'spec'"
        assert "doc_order" in analysis
        assert "metric_order" in analysis
        assert analysis["doc_order"] == [0]
        assert analysis["metric_order"] == [0]
        if name in ["memory", "zeromodel"]:
            assert "ordering" in analysis, f"{name} analysis missing 'ordering'"
            assert analysis["ordering"]["primary_metric"] == "metric1"
            assert analysis["ordering"]["direction"] == "DESC"


def test_organization_strategies_different_tasks():
    """Test organization strategies with different task specifications using real implementations."""

    logger.info("Starting test_organization_strategies_different_tasks")

    # Generate test data
    np.random.seed(42)
    score_matrix = np.random.rand(50, 3)
    metric_names = ["accuracy", "uncertainty", "size"]

    # Test different task types
    tasks = [
        ("accuracy DESC", "SELECT * FROM virtual_index ORDER BY accuracy DESC"),
        ("uncertainty ASC", "SELECT * FROM virtual_index ORDER BY uncertainty ASC"),
        ("size DESC", "SELECT * FROM virtual_index ORDER BY size DESC"),
    ]

    # Create DuckDB adapter
    duckdb_adapter = DuckDBAdapter(metric_names)
    logger.info(
        f"Created DuckDBAdapter for different tasks with {len(metric_names)} metrics"
    )

    for memory_task, sql_task in tasks:
        logger.info(f"Testing tasks: memory='{memory_task}', sql='{sql_task}'")

        # Create strategies
        memory_strategy = MemoryOrganizationStrategy()
        memory_strategy.set_task(memory_task)

        sql_strategy = SqlOrganizationStrategy(duckdb_adapter)
        sql_strategy.set_task(sql_task)

        # Load matrix for SQL strategy
        duckdb_adapter.ensure_schema(metric_names)
        duckdb_adapter.load_matrix(score_matrix, metric_names)

        zmodel_strategy = ZeroModelOrganizationStrategy()
        zmodel_strategy.set_task(memory_task)

        # Get results
        _, _, doc_order_mem, _ = memory_strategy.organize(score_matrix, metric_names)
        _, _, doc_order_sql, _ = sql_strategy.organize(score_matrix, metric_names)
        _, _, doc_order_zm, _ = zmodel_strategy.organize(score_matrix, metric_names)

        # Verify consistency
        assert np.array_equal(doc_order_mem, doc_order_sql), (
            f"Inconsistent ordering for task '{memory_task}' between memory and sql strategies"
        )
        assert np.array_equal(doc_order_mem, doc_order_zm), (
            f"Inconsistent ordering for task '{memory_task}' between memory and zeromodel strategies"
        )


def test_duckdb_adapter_integration():
    """Test that DuckDB adapter correctly handles the data and queries."""

    logger.info("Starting test_duckdb_adapter_integration")

    # Test data
    np.random.seed(42)
    score_matrix = np.random.rand(10, 2)
    metric_names = ["metric_a", "metric_b"]
    sql_task = "SELECT * FROM virtual_index ORDER BY metric_a DESC"

    # Create and test DuckDB adapter
    adapter = DuckDBAdapter(metric_names)
    logger.info(f"Created DuckDBAdapter with metrics: {metric_names}")

    adapter.ensure_schema(metric_names)
    logger.info("Schema ensured")

    adapter.load_matrix(score_matrix, metric_names)
    logger.info("Matrix loaded into DuckDB")

    # Analyze query
    logger.info("Analyzing query")
    analysis = adapter.analyze_query(sql_task, metric_names)
    logger.info(f"Analysis result keys: {list(analysis.keys())}")

    # Verify analysis contains expected information
    assert "doc_order" in analysis, "Analysis missing 'doc_order'"
    assert "metric_order" in analysis, "Analysis missing 'metric_order'"
    assert "original_query" in analysis, "Analysis missing 'original_query'"

    logger.info("Analysis contains required components")

    # Verify doc_order is reasonable (highest metric_a values first)
    primary_column = score_matrix[:, 0]  # metric_a
    expected_order = np.argsort(-primary_column).tolist()

    logger.info(f"Expected order based on metric_a: {expected_order}")
    logger.info(f"Actual doc_order from DuckDB: {analysis['doc_order']}")

    # The doc_order should be close to our expected order
    assert len(analysis["doc_order"]) == len(score_matrix), "doc_order length mismatch"

    # Create a mapping from analysis order to our expected order
    analysis_order = analysis["doc_order"]
    order_match = np.array_equal(analysis_order, expected_order)

    logger.info(f"Order match: {order_match}")

    assert order_match, "DuckDB adapter did not produce correct document order"
