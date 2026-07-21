param(
    [Parameter(Mandatory = $true)][string]$DatabasePath,
    [Parameter(Mandatory = $true)][string]$AccessId,
    [string]$Python = "python"
)

$script = @"
import json
from pathlib import Path
from zeromodel.db.runtime import build_sqlite_runtime
from zeromodel.domains.video_action_set.final_reconstruction import reconstruct_final_access_ledger

path = Path(r'''$DatabasePath''').resolve()
runtime = build_sqlite_runtime(path.as_uri().replace('file:///', 'sqlite:///'))
payload = reconstruct_final_access_ledger(runtime.video_action_set.engine.final_access_service.store, r'''$AccessId''')
print(json.dumps(payload, indent=2, sort_keys=True))
"@

& $Python -c $script
exit $LASTEXITCODE
