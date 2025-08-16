# zeromodel/duckdb_adapter.py

import logging
from typing import Any, Dict, List, Optional

import duckdb
import numpy as np
import pyarrow as pa

logger = logging.getLogger(__name__)


class DuckDBAdapter:
    def __init__(self, metric_names: List[str]):
        self._conn = duckdb.connect(database=":memory:")
        # helpful pragmas; tune if you like
        self._conn.execute("PRAGMA threads=8")
        self._conn.execute("PRAGMA memory_limit='1GB'")
        self._metric_names: List[str] = list(metric_names)
        self._matrix: Optional[np.ndarray] = None
        self._registered = False

    # ---------------- Public API -----------------------
    def ensure_schema(self, metric_names: List[str]):
        # No CREATE TABLE; just remember names for column order
        if list(metric_names) != self._metric_names:
            self._metric_names = list(metric_names)
            # Re-register on next load
            self._registered = False

    def load_matrix(self, matrix: np.ndarray, metric_names: List[str]):
        self._matrix = matrix
        self._metric_names = list(metric_names)
        n = matrix.shape[0]
        arrays = [pa.array(np.arange(n, dtype=np.int32), type=pa.int32())]
        names = ["row_id"]
        for j, name in enumerate(self._metric_names):
            arrays.append(pa.array(matrix[:, j], type=pa.float32()))
            names.append(name)
        table = pa.Table.from_arrays(arrays, names=names)
        self._conn.unregister("virtual_index")
        self._conn.register("virtual_index", table)
        self._registered = True
        logger.info(
            "DuckDB registered via Arrow: rows=%d cols=%d",
            matrix.shape[0],
            matrix.shape[1] + 1,
        )

    def analyze_query(self, sql_query: str, metric_names: List[str]) -> Dict[str, Any]:
        """
        Rewrite to only project row_id to avoid wide copies, then run it
        against the registered relation. Prefers fetchnumpy() to avoid
        hard deps on pyarrow/pandas.
        """
        if not self._registered:
            raise RuntimeError("No registered relation; call load_matrix() first.")

        q = sql_query.strip()
        if q.upper().startswith("SELECT *"):
            q = q.replace("SELECT *", "SELECT row_id", 1)
        else:
            q = f"SELECT row_id FROM ({sql_query}) AS user_sorted_view"

        logger.debug("DuckDB query: %s", q)
        cur = self._conn.execute(q)

        # 1) Fast path: NumPy (no pyarrow/pandas required)
        try:
            npres = cur.fetchnumpy()  # returns dict[str, np.ndarray]
            idx = npres["row_id"].astype(np.int32, copy=False)
            return {
                "doc_order": idx.tolist(),
                "metric_order": list(range(len(metric_names))),
                "original_query": sql_query,
            }
        except Exception as e_np:
            logger.debug(
                "fetchnumpy() unavailable/failed, trying Arrow then pandas: %s", e_np
            )

        # 2) Arrow fallback (requires pyarrow)
        try:
            arr_tbl = cur.arrow()
      
            idx = np.array(arr_tbl.column("row_id"), copy=False).astype(
                np.int32, copy=False
            )
            return {
                "doc_order": idx.tolist(),
                "metric_order": list(range(len(metric_names))),
                "original_query": sql_query,
            }
        except Exception as e_arrow:
            logger.debug("Arrow fetch failed: %s", e_arrow)

        # 3) Pandas fallback (requires pandas)
        try:
            df = cur.df()
            idx = df["row_id"].to_numpy(dtype="int32", copy=False)
            return {
                "doc_order": idx.tolist(),
                "metric_order": list(range(len(metric_names))),
                "original_query": sql_query,
            }
        except Exception as e_pd:
            raise RuntimeError(
                f"Unable to fetch query result via NumPy/Arrow/pandas. "
                f"Install pyarrow or pandas, or upgrade duckdb. Last error: {e_pd}"
            ) from e_pd

    @property
    def connection(self):  # Expose if low-level access needed
        return self._conn
