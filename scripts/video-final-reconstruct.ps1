param(
    [Parameter(Mandatory = $true)][string]$DatabasePath,
    [Parameter(Mandatory = $true)][string]$AccessId,
    [string]$Python = "python"
)

& $Python -m zeromodel.persistence.sqlalchemy.video_action_set_final_admin_cli reconstruct `
    --database-path $DatabasePath `
    --access-id $AccessId
exit $LASTEXITCODE
