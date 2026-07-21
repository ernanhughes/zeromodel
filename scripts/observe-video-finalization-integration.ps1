[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ValidationRoot
)

$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath($ValidationRoot)
$temporaryRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$forbidden = [IO.Path]::GetFullPath("C:\Projects\zeromodel-stage8")
function Test-PathWithin {
    param(
        [Parameter(Mandatory = $true)][string]$Child,
        [Parameter(Mandatory = $true)][string]$Parent
    )
    $childPath = [IO.Path]::GetFullPath($Child).TrimEnd('\')
    $parentPath = [IO.Path]::GetFullPath($Parent).TrimEnd('\')
    return $childPath.Equals($parentPath, [StringComparison]::OrdinalIgnoreCase) -or
        $childPath.StartsWith($parentPath + '\', [StringComparison]::OrdinalIgnoreCase)
}

if (-not (Test-PathWithin -Child $root -Parent $temporaryRoot)) {
    throw "ValidationRoot must be under the system temporary directory."
}
if (Test-PathWithin -Child $root -Parent $forbidden) {
    throw "The Stage 8 repository is forbidden."
}
$marker = Join-Path $root ".zeromodel-synthetic-validation"
if (-not (Test-Path -LiteralPath $marker -PathType Leaf)) {
    throw "ValidationRoot is not a bounded synthetic validation directory."
}

$statePath = Join-Path $root "run-state.json"
$summaryJsonPath = Join-Path $root "summary.json"
$summaryMarkdownPath = Join-Path $root "summary.md"
$state = if (Test-Path -LiteralPath $statePath -PathType Leaf) {
    Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
} else {
    $null
}
$runnerAlive = $false
$groupAlive = $false
if ($null -ne $state) {
    $runnerAlive = $null -ne (Get-Process -Id $state.runner_pid -ErrorAction SilentlyContinue)
    if ($null -ne $state.group_pid) {
        $groupAlive = $null -ne (Get-Process -Id $state.group_pid -ErrorAction SilentlyContinue)
    }
}
$logs = @(
    Get-ChildItem -LiteralPath (Join-Path $root "logs") -File -ErrorAction SilentlyContinue |
        Sort-Object Name |
        ForEach-Object {
            [ordered]@{
                name = $_.Name
                bytes = $_.Length
                last_write_utc = $_.LastWriteTimeUtc.ToString("o")
            }
        }
)

[ordered]@{
    version = "zeromodel-video-finalization-integration-observation/v1"
    observed_utc = [DateTime]::UtcNow.ToString("o")
    validation_root = $root
    run_state = $state
    runner_alive = $runnerAlive
    group_alive = $groupAlive
    summary_json_present = Test-Path -LiteralPath $summaryJsonPath -PathType Leaf
    summary_markdown_present = Test-Path -LiteralPath $summaryMarkdownPath -PathType Leaf
    logs = $logs
} | ConvertTo-Json -Depth 10
