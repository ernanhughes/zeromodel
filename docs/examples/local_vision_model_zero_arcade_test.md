# Local Gemma 4 → ZeroModel Arcade Test

This is a **local-only experiment**. Do not add it to GitHub Actions.

## Pipeline

```text
rendered arcade frame
        ↓
local Gemma 4 through Ollama
        ↓
strict state JSON
        ↓
ZeroModel policy row
        ↓
VPMPolicyLookup
        ↓
LEFT / RIGHT / STAY / FIRE
```

Gemma receives no policy rules and is never asked to choose an action.

## Setup

From the current ZeroModel repository root:

```powershell
git switch main
git pull --ff-only
python -m pip install -r requirements-dev.txt
python -m pip install pillow
ollama list
```

Copy `local_gemma4_zero_arcade_test.py` into the repository root or a scratch
folder. Use the exact Gemma tag shown by `ollama list`.

## 1. Verify the harness without Gemma

```powershell
python .\local_gemma4_zero_arcade_test.py `
  --backend fake `
  --mode smoke `
  --render labelled
```

Expected: 8 accepted, 8 exact states, 8 correct actions.

## 2. First real Gemma test

```powershell
python .\local_gemma4_zero_arcade_test.py `
  --backend ollama `
  --model gemma4:e4b `
  --mode smoke `
  --render labelled
```

Replace `gemma4:e4b` with your installed tag.

## 3. Harder unlabelled test

```powershell
python .\local_gemma4_zero_arcade_test.py `
  --backend ollama `
  --model gemma4:e4b `
  --mode smoke `
  --render unlabelled
```

## 4. Complete 112-state test

Run this only after the smoke test is sensible:

```powershell
python .\local_gemma4_zero_arcade_test.py `
  --backend ollama `
  --model gemma4:e4b `
  --mode all `
  --render labelled `
  --confidence-threshold 0.70
```

## Send back

Each run creates:

```text
local-results/gemma4-zero-arcade-<timestamp>/
├── images/
├── cases.jsonl
├── summary.json
└── run-manifest.json
```

Send back `summary.json` and `cases.jsonl`. The results separate exact state
accuracy from action-equivalent accuracy and show errors by tank column, target
presence, target column, cooldown, parsing, confidence, and latency.
