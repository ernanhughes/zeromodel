$out = "build\visual-phase-one-local"
New-Item -ItemType Directory -Force $out | Out-Null

$command = @"
python examples/arcade_visual_address_benchmark.py --encoder dinov2 --device $device --local-files-only --variants-per-family 3 --include-traces --output-dir $out
"@

$command | Set-Content -Encoding utf8 "$out\command.txt"
$env:ZEROMODEL_BENCHMARK_COMMAND = $command

$started = Get-Date

python examples/arcade_visual_address_benchmark.py `
  --encoder dinov2 `
  --device cuda `
  --local-files-only `
  --variants-per-family 3 `
  --include-traces `
  --output-dir $out `
  2>&1 | Tee-Object -FilePath "$out\dino-full-run.log"

$exitCode = $LASTEXITCODE
$finished = Get-Date

@"
started_utc=$($started.ToUniversalTime().ToString("o"))
finished_utc=$($finished.ToUniversalTime().ToString("o"))
elapsed_seconds=$(($finished - $started).TotalSeconds)
exit_code=$exitCode
device=$device
"@ | Set-Content -Encoding utf8 "$out\runtime.txt"

if ($exitCode -ne 0) {
    throw "Full DINO benchmark failed with exit code $exitCode"
}