$Version = "1.1.0"

$Distributions = @(
    "zeromodel",
    "zeromodel-analysis",
    "zeromodel-observation",
    "zeromodel-perception",
    "zeromodel-vision",
    "zeromodel-video",
    "zeromodel-sqlalchemy",
    "zeromodel-artifacts",
    "zeromodel-trust",
    "zeromodel-navigation"
)

$Artifacts = @(
    Get-ChildItem .\packages -Recurse -File |
        Where-Object {
            $_.Directory.Name -eq "dist" -and
            (
                $_.Name -like "*-$Version-*.whl" -or
                $_.Name -like "*-$Version.tar.gz"
            )
        } |
        Sort-Object Name
)

if ($Artifacts.Count -ne 20) {
    throw (
        "Expected 20 local artifacts for version {0}; found {1}." -f
        $Version,
        $Artifacts.Count
    )
}

$PublishedNames = [System.Collections.Generic.HashSet[string]]::new(
    [System.StringComparer]::OrdinalIgnoreCase
)

foreach ($Distribution in $Distributions) {
    try {
        $Response = Invoke-RestMethod `
            -Uri (
                "https://pypi.org/pypi/{0}/{1}/json" -f
                $Distribution,
                $Version
            ) `
            -ErrorAction Stop

        foreach ($File in $Response.urls) {
            [void]$PublishedNames.Add($File.filename)
        }
    }
    catch {
        # Distribution/version is not published yet.
    }
}

$MissingArtifacts = @(
    $Artifacts |
        Where-Object {
            -not $PublishedNames.Contains($_.Name)
        } |
        Sort-Object Name
)

Write-Host ""
Write-Host (
    "Already published files: {0}" -f $PublishedNames.Count
) -ForegroundColor Green

Write-Host (
    "Files still missing: {0}" -f $MissingArtifacts.Count
) -ForegroundColor Yellow

$MissingArtifacts |
    Select-Object Name, FullName |
    Format-Table -AutoSize

foreach ($Artifact in $MissingArtifacts) {
    Write-Host ""
    Write-Host (
        "Uploading {0}" -f $Artifact.Name
    ) -ForegroundColor Cyan

    $Succeeded = $false

    for ($Attempt = 1; $Attempt -le 8; $Attempt++) {
        python -m twine upload `
            --repository pypi `
            --skip-existing `
            --verbose `
            $Artifact.FullName

        if ($LASTEXITCODE -eq 0) {
            $Succeeded = $true
            Write-Host (
                "Uploaded {0}" -f $Artifact.Name
            ) -ForegroundColor Green
            break
        }

        $Delay = 60 * $Attempt

        Write-Host (
            "Attempt {0} failed. Waiting {1} seconds." -f
            $Attempt,
            $Delay
        ) -ForegroundColor Yellow

        Start-Sleep -Seconds $Delay
    }

    if (-not $Succeeded) {
        throw (
            "Upload failed repeatedly for {0}." -f
            $Artifact.Name
        )
    }

    Start-Sleep -Seconds 45
}