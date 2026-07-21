param(
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [Parameter(Mandatory = $true)][string]$AuthorizationFile,
    [Parameter(Mandatory = $true)][string]$ExpectedAuthorizationDigest,
    [Parameter(Mandatory = $true)][string]$ExpectedSealedPlanDigest,
    [Parameter(Mandatory = $true)][string]$DatabasePath,
    [string]$Python = "python"
)

& $Python -m zeromodel.video_action_set_final_cli `
    --output-dir $OutputDir `
    --authorization-file $AuthorizationFile `
    --expected-authorization-digest $ExpectedAuthorizationDigest `
    --expected-sealed-plan-digest $ExpectedSealedPlanDigest `
    --database-path $DatabasePath `
    --preflight-only

exit $LASTEXITCODE
