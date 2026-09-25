function Start-Training {
    param(
        [Parameter(Mandatory=$true)][string]$Dataset,
        [string]$Run = $Dataset,
        [string]$User = "AdamAxelrod",
        [int]$Steps = 50000,
        [int]$BatchSize = 32,
        [int]$NumWorkers = 4,
        [int]$SaveFreq = 5000,
        [int]$ChunkSize = 50,
        [Nullable[int]]$NActionSteps = $null,
        [string]$PolicyType = "act",
        [switch]$NoWandb,
        [switch]$NoHubPush,
        [switch]$NoAmp
    )
    $repo = "$User/$Dataset"
    $modelRepo = "$User/${Dataset}_model"

    # Auto-suffix output_dir if it already exists, so re-runs don't collide.
    $baseOutput = "outputs\train\$Run"
    $outputDir = $baseOutput
    $suffix = 2
    while (Test-Path $outputDir) {
        $outputDir = "${baseOutput}_$suffix"
        $suffix++
    }

    $runTag = $outputDir | Split-Path -Leaf
    $logFile = "training_log_$runTag.txt"

    # n_action_steps defaults to chunk_size (full-chunk inference, original ACT behavior).
    if ($null -eq $NActionSteps) { $NActionSteps = $ChunkSize }

    $args = @(
        "--dataset.repo_id=$repo"
        "--policy.type=$PolicyType"
        "--output_dir=$outputDir"
        "--policy.device=cuda"
        "--policy.chunk_size=$ChunkSize"
        "--policy.n_action_steps=$NActionSteps"
        "--batch_size=$BatchSize"
        "--num_workers=$NumWorkers"
        "--save_checkpoint=true"
        "--save_freq=$SaveFreq"
        "--job_name=$runTag"
        "--steps=$Steps"
    )
    if (-not $NoAmp)     { $args += "--policy.use_amp=true" }
    if (-not $NoWandb)   { $args += "--wandb.enable=true" }
    if (-not $NoHubPush) {
        $args += "--policy.push_to_hub=true"
        $args += "--policy.repo_id=$modelRepo"
    }

    Write-Host "Dataset:     $repo"
    Write-Host "Output dir:  $outputDir"
    Write-Host "Log file:    $logFile"
    Write-Host "Policy:      $PolicyType (chunk_size=$ChunkSize, n_action_steps=$NActionSteps)"
    Write-Host "Batch x WK:  $BatchSize x $NumWorkers"
    Write-Host "Steps:       $Steps (save every $SaveFreq)"
    Write-Host ""

    # Silence the torchvision video-decoding deprecation warning for the duration of this call.
    $prevWarnings = $env:PYTHONWARNINGS
    $env:PYTHONWARNINGS = "ignore::UserWarning:torchvision.io._video_deprecation_warning"
    try {
        # ForEach-Object { "$_" } unwraps the NativeCommandError objects PS 5.1 wraps
        # around stderr lines, so INFO logs render normally instead of red.
        lerobot-train @args 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $logFile
    } finally {
        $env:PYTHONWARNINGS = $prevWarnings
    }
}

# . .\train.ps1
# Start-Training -Dataset "red_dot_black_pointer" -Steps 50000 -BatchSize 32