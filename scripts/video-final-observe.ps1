param(
    [Parameter(Mandatory = $true)][string]$DatabasePath,
    [Parameter(Mandatory = $true)][string]$AccessId,
    [string]$Python = "python"
)

& $Python -m zeromodel.video_action_set_final_admin_cli observe `
    --database-path $DatabasePath `
    --access-id $AccessId
exit $LASTEXITCODE
