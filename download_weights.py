from huggingface_hub import login, snapshot_download

login()

snapshot_download(
    repo_id="mateuszpach/sae-for-vlm",
    repo_type="model",
    local_dir="checkpoints_dir",
)
