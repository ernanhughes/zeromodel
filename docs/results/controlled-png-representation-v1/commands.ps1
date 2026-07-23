# Run from the ZeroModel repository root in PowerShell.

$BaseUrl = "http://localhost:11434"
$Model = "qwen3.5:latest"
$Seed = 0
$TimeoutSeconds = 180
$ConfidenceThreshold = 0.0
$SmokeRoot = ".\local-results\controlled-png-representation-v1-smoke-20260723T220702Z"
$SmokeDatabase = Join-Path $SmokeRoot "benchmark.sqlite"

# Baseline: 16 real-provider calls.
python .\examples\arcade_png_representation_benchmark.py `
  --backend ollama `
  --model $Model `
  --base-url $BaseUrl `
  --fixture smoke `
  --variants labelled-v1,unlabelled-v1 `
  --store sqlite `
  --sqlite-path $SmokeDatabase `
  --output-dir $SmokeRoot `
  --compile-reports `
  --seed $Seed `
  --timeout $TimeoutSeconds `
  --confidence-threshold $ConfidenceThreshold `
  --write-pngs

# Isolated interventions: reuses the two baseline runs and makes 32 new calls.
python .\examples\arcade_png_representation_benchmark.py `
  --backend ollama `
  --model $Model `
  --base-url $BaseUrl `
  --fixture smoke `
  --variants labelled-v1,unlabelled-v1,cooldown-shape-v1,cooldown-dual-v1,cooldown-redundant-v1,lane-enhanced-v1 `
  --store sqlite `
  --sqlite-path $SmokeDatabase `
  --output-dir $SmokeRoot `
  --compile-reports `
  --seed $Seed `
  --timeout $TimeoutSeconds `
  --confidence-threshold $ConfidenceThreshold `
  --write-pngs `
  --resume
