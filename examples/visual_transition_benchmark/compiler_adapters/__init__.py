"""Domain adapters for the evidence contract compiler.

Each adapter declares candidate regions and ``VisualEvidenceRequirement``s for
its domain's benchmark cases, and builds development/evaluation samples from
the domain's existing dataset generation (``dataset.py`` / ``faults.py``).
Ground-truth ``true_before``/``true_after`` values are privileged and used
only for scoring -- never passed into the compiler's candidate decoders.
"""
