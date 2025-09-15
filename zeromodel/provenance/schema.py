#  zeromodel/provenance/schema.py
from __future__ import annotations

import base64
import hashlib
import json
import struct
import zlib
from dataclasses import asdict, dataclass, field
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from zeromodel.provenance.metadata import VPF_FOOTER_MAGIC, VPF_MAGIC_HEADER
from zeromodel.provenance.png_text import png_read_text_chunk, png_write_text_chunk

VPF_VERSION = "1.0"

# --- Schema ------------------------------------------------------------------


@dataclass
class VPFPipeline:
    graph_hash: str = ""
    step: str = ""
    step_schema_hash: str = ""


@dataclass
class VPFModel:
    id: str = ""
    assets: Dict[str, str] = field(default_factory=dict)


@dataclass
class VPFDeterminism:
    seed_global: int = 0
    seed_sampler: int = 0
    rng_backends: List[str] = field(default_factory=list)


@dataclass
class VPFParams:
    sampler: str = ""
    steps: int = 0
    cfg_scale: float = 0.0
    size: List[int] = field(default_factory=lambda: [0, 0])
    preproc: List[str] = field(default_factory=list)
    postproc: List[str] = field(default_factory=list)
    stripe: Optional[Dict[str, Any]] = None


@dataclass
class VPFInputs:
    prompt: str = ""
    negative_prompt: Optional[str] = None
    prompt_hash: str = ""  # tests may supply prompt_sha3 instead; we’ll map
    image_refs: List[str] = field(default_factory=list)
    retrieved_docs_hash: Optional[str] = None
    task: str = ""


@dataclass
class VPFMetrics:
    aesthetic: float = 0.0
    coherence: float = 0.0
    safety_flag: float = 0.0


@dataclass
class VPFLineage:
    parents: List[str] = field(default_factory=list)
    content_hash: str = ""  # tests may set later / or left empty
    vpf_hash: str = ""  # will be computed on serialize


@dataclass
class VPFAuth:
    algo: str = ""
    pubkey: str = ""
    sig: str = ""


@dataclass
class VPF:
    vpf_version: str = "1.0"
    pipeline: VPFPipeline = field(default_factory=VPFPipeline)
    model: VPFModel = field(default_factory=VPFModel)
    determinism: VPFDeterminism = field(default_factory=VPFDeterminism)
    params: VPFParams = field(default_factory=VPFParams)
    inputs: VPFInputs = field(default_factory=VPFInputs)
    metrics: VPFMetrics = field(default_factory=VPFMetrics)
    lineage: VPFLineage = field(default_factory=VPFLineage)
    signature: Optional[VPFAuth] = None



