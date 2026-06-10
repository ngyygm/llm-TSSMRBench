$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$python = if ($env:BTQ_PYTHON) { $env:BTQ_PYTHON } else { 'python' }
$config = Join-Path $root 'benchmark\configs\state_version_experiment_config_deepseek_flash_memory.yaml'
$merged = Join-Path $root 'benchmark\data\prototypes\github_release_note_v2\formal_300repo_unified_v1\official_300_merged.json'
$script = Join-Path $root 'benchmark\scripts\82_run_merged_github_release_unified_global_pool_evaluation.py'

$runTag = 'fullrun_v2'
$logDir = Join-Path $root ("benchmark\logs\formal_globalpool_" + $runTag)
$resultRoot = Join-Path $root 'benchmark\data\prototype_eval_results'
$mem0Out = Join-Path $resultRoot ("official_300repo_release_unified_v1_mem0_deepseekflash_globalpool_taskk_" + $runTag)
$graphitiOut = Join-Path $resultRoot ("official_300repo_release_unified_v1_graphiti_deepseekflash_globalpool_taskk_" + $runTag)
$mem0Log = Join-Path $logDir 'mem0_globalpool.log'
$graphitiLog = Join-Path $logDir 'graphiti_globalpool.log'
$launchMeta = Join-Path $logDir 'launch_meta.json'

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
New-Item -ItemType Directory -Force -Path $mem0Out | Out-Null
New-Item -ItemType Directory -Force -Path $graphitiOut | Out-Null

$meta = [ordered]@{
    started_at = (Get-Date).ToString('s')
    run_tag = $runTag
    python = $python
    script = $script
    config = $config
    merged_json = $merged
    mem0_output_dir = $mem0Out
    graphiti_output_dir = $graphitiOut
    max_workers = 50
    save_every = 50
}
$meta | ConvertTo-Json -Depth 4 | Set-Content -Path $launchMeta -Encoding UTF8

Add-Content -Path $mem0Log -Value ("`n=== MEM0 START " + (Get-Date).ToString('s') + " ===")
& $python -B $script `
  --config $config `
  --merged-json $merged `
  --system mem0 `
  --output-dir $mem0Out `
  --max-workers 50 `
  --save-every 50 `
  *>> $mem0Log
Add-Content -Path $mem0Log -Value ("=== MEM0 END " + (Get-Date).ToString('s') + " ===")

Add-Content -Path $graphitiLog -Value ("`n=== GRAPHITI START " + (Get-Date).ToString('s') + " ===")
& $python -B $script `
  --config $config `
  --merged-json $merged `
  --system graphiti `
  --output-dir $graphitiOut `
  --max-workers 50 `
  --save-every 50 `
  *>> $graphitiLog
Add-Content -Path $graphitiLog -Value ("=== GRAPHITI END " + (Get-Date).ToString('s') + " ===")
