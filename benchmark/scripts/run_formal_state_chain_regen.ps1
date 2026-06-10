param(
    [Parameter(Mandatory = $true)]
    [string]$GitHubToken,
    [switch]$SkipNarrative,
    [switch]$SkipGithubTrain,
    [switch]$SkipGithubDevTest
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$python = if ($env:BTQ_PYTHON) { $env:BTQ_PYTHON } else { "python" }
$logsRoot = Join-Path $repoRoot "benchmark\logs\background_runs"

New-Item -ItemType Directory -Force -Path $logsRoot | Out-Null
$env:GITHUB_TOKEN = $GitHubToken

function New-RunnerScript {
    param(
        [string]$Path,
        [string[]]$Commands
    )
    $lines = @('$ErrorActionPreference = "Stop"')
    foreach ($cmd in $Commands) {
        $lines += $cmd
        $lines += 'if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }'
    }
    Set-Content -LiteralPath $Path -Value ($lines -join "`r`n") -Encoding UTF8
}

$narrativeRunner = Join-Path $logsRoot "formal_narrative_runner.ps1"
$githubTrainRunner = Join-Path $logsRoot "formal_github_train_runner.ps1"
$githubDevTestRunner = Join-Path $logsRoot "formal_github_devtest_runner.ps1"

New-RunnerScript -Path $narrativeRunner -Commands @(
    "& '$python' '$repoRoot\benchmark\scripts\59_generate_narrative_summary_specs_from_catalog.py' --phase formal --split train --resume --max-consecutive-failures 3",
    "& '$python' '$repoRoot\benchmark\scripts\59_generate_narrative_summary_specs_from_catalog.py' --phase formal --split dev --resume --max-consecutive-failures 3",
    "& '$python' '$repoRoot\benchmark\scripts\59_generate_narrative_summary_specs_from_catalog.py' --phase formal --split test --resume --max-consecutive-failures 3",
    "& '$python' '$repoRoot\benchmark\scripts\58_prepare_narrative_summary_source_bundles.py' --phase formal --split train --resume --update-manifest",
    "& '$python' '$repoRoot\benchmark\scripts\58_prepare_narrative_summary_source_bundles.py' --phase formal --split dev --resume --update-manifest",
    "& '$python' '$repoRoot\benchmark\scripts\58_prepare_narrative_summary_source_bundles.py' --phase formal --split test --resume --update-manifest",
    "& '$python' '$repoRoot\benchmark\scripts\52_generate_state_chains.py' --phase formal --domain narrative_evolution --split train --resume --update-manifest --max-consecutive-failures 3",
    "& '$python' '$repoRoot\benchmark\scripts\52_generate_state_chains.py' --phase formal --domain narrative_evolution --split dev --resume --update-manifest --max-consecutive-failures 3",
    "& '$python' '$repoRoot\benchmark\scripts\52_generate_state_chains.py' --phase formal --domain narrative_evolution --split test --resume --update-manifest --max-consecutive-failures 3"
)

New-RunnerScript -Path $githubTrainRunner -Commands @(
    "& '$python' '$repoRoot\benchmark\scripts\54_collect_github_raw_artifacts.py' --phase formal --split train --repo-file '$repoRoot\benchmark\configs\github_formal_train_repos.txt' --issues-per-repo 5 --issue-fetch-limit 60 --min-issue-comments 6 --min-issue-body-chars 250 --max-total-bundles 120 --resume",
    "& '$python' '$repoRoot\benchmark\scripts\53_prepare_github_source_bundles.py' --input '$repoRoot\benchmark\data\state_chain_bench\en\formal\github_evolution\train\raw_github_artifacts.jsonl' --phase formal --split train --resume --update-manifest",
    "& '$python' '$repoRoot\benchmark\scripts\52_generate_state_chains.py' --phase formal --domain github_evolution --split train --resume --update-manifest --max-consecutive-failures 3"
)

New-RunnerScript -Path $githubDevTestRunner -Commands @(
    "& '$python' '$repoRoot\benchmark\scripts\54_collect_github_raw_artifacts.py' --phase formal --split dev --repo-file '$repoRoot\benchmark\configs\github_formal_dev_repos.txt' --issues-per-repo 5 --issue-fetch-limit 60 --min-issue-comments 6 --min-issue-body-chars 250 --max-total-bundles 40 --resume",
    "& '$python' '$repoRoot\benchmark\scripts\53_prepare_github_source_bundles.py' --input '$repoRoot\benchmark\data\state_chain_bench\en\formal\github_evolution\dev\raw_github_artifacts.jsonl' --phase formal --split dev --resume --update-manifest",
    "& '$python' '$repoRoot\benchmark\scripts\52_generate_state_chains.py' --phase formal --domain github_evolution --split dev --resume --update-manifest --max-consecutive-failures 3",
    "& '$python' '$repoRoot\benchmark\scripts\54_collect_github_raw_artifacts.py' --phase formal --split test --repo-file '$repoRoot\benchmark\configs\github_formal_test_repos.txt' --issues-per-repo 5 --issue-fetch-limit 60 --min-issue-comments 6 --min-issue-body-chars 250 --max-total-bundles 40 --resume",
    "& '$python' '$repoRoot\benchmark\scripts\53_prepare_github_source_bundles.py' --input '$repoRoot\benchmark\data\state_chain_bench\en\formal\github_evolution\test\raw_github_artifacts.jsonl' --phase formal --split test --resume --update-manifest",
    "& '$python' '$repoRoot\benchmark\scripts\52_generate_state_chains.py' --phase formal --domain github_evolution --split test --resume --update-manifest --max-consecutive-failures 3"
)

$jobs = @()
if (-not $SkipNarrative) {
    $jobs += @{
        Name = "formal_narrative"
        Script = $narrativeRunner
        Out = Join-Path $logsRoot "formal_narrative.out.log"
        Err = Join-Path $logsRoot "formal_narrative.err.log"
    }
}
if (-not $SkipGithubTrain) {
    $jobs += @{
        Name = "formal_github_train"
        Script = $githubTrainRunner
        Out = Join-Path $logsRoot "formal_github_train.out.log"
        Err = Join-Path $logsRoot "formal_github_train.err.log"
    }
}
if (-not $SkipGithubDevTest) {
    $jobs += @{
        Name = "formal_github_devtest"
        Script = $githubDevTestRunner
        Out = Join-Path $logsRoot "formal_github_devtest.out.log"
        Err = Join-Path $logsRoot "formal_github_devtest.err.log"
    }
}

$started = @()
foreach ($job in $jobs) {
    $proc = Start-Process -FilePath "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" `
        -ArgumentList @("-ExecutionPolicy", "Bypass", "-File", $job.Script) `
        -WindowStyle Hidden `
        -RedirectStandardOutput $job.Out `
        -RedirectStandardError $job.Err `
        -PassThru
    $started += [pscustomobject]@{
        name = $job.Name
        pid = $proc.Id
        script = $job.Script
        stdout = $job.Out
        stderr = $job.Err
    }
}

$started | ConvertTo-Json -Depth 3
