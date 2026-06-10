$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$python = if ($env:BTQ_PYTHON) { $env:BTQ_PYTHON } else { 'python' }
$config = Join-Path $root 'benchmark\configs\state_version_experiment_config_deepseek_flash_memory.yaml'
$merged = Join-Path $root 'benchmark\data\prototypes\github_release_note_v2\formal_300repo_unified_v1\official_300_merged.json'
$script = Join-Path $root 'benchmark\scripts\82_run_merged_github_release_unified_global_pool_evaluation.py'
$logDir = Join-Path $root 'benchmark\logs\formal_globalpool'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$mem0Out = Join-Path $root 'benchmark\data\prototype_eval_results\official_300repo_release_unified_v1_mem0_deepseekflash_globalpool_taskk'
$graphitiOut = Join-Path $root 'benchmark\data\prototype_eval_results\official_300repo_release_unified_v1_graphiti_deepseekflash_globalpool_taskk'
$mem0Log = Join-Path $logDir 'mem0_globalpool.log'
$graphitiLog = Join-Path $logDir 'graphiti_globalpool.log'

& $python -B $script `
  --config $config `
  --merged-json $merged `
  --system mem0 `
  --output-dir $mem0Out `
  --max-workers 50 `
  --save-every 50 `
  *>> $mem0Log

& $python -B $script `
  --config $config `
  --merged-json $merged `
  --system graphiti `
  --output-dir $graphitiOut `
  --max-workers 50 `
  --save-every 50 `
  *>> $graphitiLog
