from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.video_action_set.video_action_set_cli import OUTPUT_DIR, main  # noqa: E402, F401


if __name__ == "__main__":
    main()
