from __future__ import annotations

from pathlib import Path

from scripts._validate_release_candidate_impl import *  # noqa: F403

PACKAGES["perception"] = {  # noqa: F405
    "path": Path("packages/perception"),
    "distribution": "zeromodel-perception",
    "wheel_stem": "zeromodel_perception",
    "namespace": "zeromodel.perception",
    "requires": {
        "numpy>=1.23",
        "pillow>=9.0",
        f"zeromodel=={VERSION}",  # noqa: F405
        f"zeromodel-observation=={VERSION}",  # noqa: F405
    },
    "depends_on": ("core", "observation"),
}


if __name__ == "__main__":
    main()  # noqa: F405
