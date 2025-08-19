# zeromodel/pipeline/core.py
"""
ZeroModel Pipeline Core

High-level design:
- A Pipeline is a lightweight orchestrator that threads an immutable (data, metadata) pair
  through a sequence of focused PipelineStage objects.
- Each PipelineStage implements a single transformation, appending provenance and diagnostics.
- Intelligence is expressed through structural reorganization and provenance capture rather
  than heavy computation inside individual stages.

Key concepts:
- PipelineContext: Immutable carrier (data + metadata); new metadata merged per stage.
- PipelineStage: Abstract base; concrete stages implement validate_params() and process().
- Provenance: Accumulated lightweight trace of stage executions (name, params, timestamp).

Extension guidelines:
1. Keep stages pure (avoid mutating input arrays in-place unless explicitly documented).
2. Use validate_params() to fail fast on misconfiguration.
3. Record only essential provenance fields (avoid large payloads).
4. Prefer returning small, composable metadata entries over deeply nested structures.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """
    Immutable context passed between stages.

    Attributes:
        data: Current data artifact (e.g., VPM ndarray or higher structure).
        metadata: Accumulated metadata / diagnostics (lightweight, JSON-friendly).

    Immutability note:
        update() returns a new PipelineContext with merged metadata, preserving prior keys.
    """
    data: Any
    metadata: Dict[str, Any]

    def update(self, **updates) -> "PipelineContext":
        """
        Produce a new context with metadata merged with provided updates.

        Returns:
            New PipelineContext (original untouched).
        """
        new_metadata = {**self.metadata, **updates}
        return PipelineContext(self.data, new_metadata)


class PipelineStage(ABC):
    """
    Abstract base for all pipeline stages.

    Responsibilities:
    - Validate configuration (validate_params)
    - Transform the VPM or associated structure (process)
    - Contribute provenance (record_provenance)

    Contract:
    - process(vpm, context) must return (new_vpm, stage_metadata)
    - Must be side-effect minimal (except for logging)
    - stage_metadata should be shallow, serializable, and merge-safe
    """

    name: str = "base"        # Unique stage name (enforced by Pipeline)
    category: str = "base"    # Optional grouping / taxonomy hint

    def __init__(self, **params):
        """
        Initialize the stage with provided parameter dictionary.

        Params:
            **params: Stage-specific configuration (validated in validate_params()).
        """
        self.params = params

    @abstractmethod
    def process(
        self,
        vpm: np.ndarray,
        context: Dict[str, Any] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform the stage transformation.

        Args:
            vpm: Input VPM array (shape is stage-dependent).
            context: Optional shared context (mutable dict) for cross-stage coordination.

        Returns:
            (transformed_vpm, stage_metadata)

        Note:
            stage_metadata is merged into the pipeline metadata at the orchestrator layer.
        """
        raise NotImplementedError

    @abstractmethod
    def validate_params(self):
        """
        Validate stage parameters.

        Should raise ValueError / TypeError on invalid configuration early.
        """
        raise NotImplementedError

    def get_context(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Ensure a valid context dict exists and contains provenance bucket.

        Args:
            context: Possibly None or an existing shared dict.

        Returns:
            Context dict with a 'provenance' list.
        """
        if context is None:
            context = {}
        if "provenance" not in context:
            context["provenance"] = []
        return context

    def record_provenance(self, context: Dict[str, Any], stage_name: str, params: Dict[str, Any]):
        """
        Append a provenance entry documenting this stage execution.

        Stored fields:
            stage: Name identifier
            params: Shallow copy of stage parameters
            timestamp: High-resolution wall-clock (NumPy datetime64)
        """
        context["provenance"].append({
            "stage": stage_name,
            "params": params,
            "timestamp": np.datetime64("now"),
        })

    # Optional: Define __call__ for ergonomic PipelineStage invocation if desired.
    # def __call__(self, ctx: PipelineContext) -> PipelineContext:
    #     vpm, meta = self.process(ctx.data, ctx.metadata)
    #     return PipelineContext(vpm, {**ctx.metadata, **meta})


class Pipeline:
    """
    Orchestrates a sequence of PipelineStage instances.

    Design notes:
    - Each stage is invoked in order; failures abort execution with logged duration.
    - Metadata accumulation is monotonic: later stages can overwrite keys intentionally.
    - Stages may (optionally) implement __call__ for direct invocation; current version
      expects a callable interface (stage(context))—extend cautiously if adding it.

    Usage pattern:
        pipeline = Pipeline([StageA(...), StageB(...), ...])
        final_data, final_meta = pipeline.run(data)

    Failure handling:
        Exceptions propagate after logging elapsed time for observability.
    """

    def __init__(self, stages: List[PipelineStage]):
        """
        Initialize with a list of pre-configured, validated stages.

        Args:
            stages: Ordered list of PipelineStage objects.
        """
        self.stages = stages
        self.validate_stages()

    def validate_stages(self):
        """
        Ensure uniqueness of stage names to avoid provenance ambiguity.

        Raises:
            ValueError if duplicate stage names detected.
        """
        names = [stage.name for stage in self.stages]
        if len(names) != len(set(names)):
            raise ValueError("All pipeline stages must have unique names")

    def run(
        self,
        initial_data: Any,
        initial_metadata: Dict[str, Any] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Execute the pipeline sequentially.

        Args:
            initial_data: Starting data artifact (e.g., raw VPM).
            initial_metadata: Optional seed metadata dict.

        Returns:
            (final_data, final_metadata)

        Logging:
            Emits start, success (with total duration), or failure with timing.

        Note:
            Stages are expected to return updated PipelineContext (call protocol).
        """
        context = PipelineContext(
            data=initial_data,
            metadata=initial_metadata or {}
        )

        logger.info(f"Starting pipeline execution with {len(self.stages)} stages")
        start_time = time.time()

        try:
            for stage in self.stages:
                # Assumes stage is callable returning a new PipelineContext
                context = stage(context)

            total_time = time.time() - start_time
            logger.info(f"Pipeline completed successfully in {total_time:.3f}s")
            return context.data, context.metadata

        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Pipeline failed after {total_time:.3f}s: {e}")
            raise