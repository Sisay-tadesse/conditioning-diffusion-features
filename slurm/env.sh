# Everything cluster-specific lives here; the job scripts source it from the
# repo root. Override any of these in the environment before submitting.
THESIS_ENV=${THESIS_ENV:-$HOME/thesis-env}   # uv venv, on zfsstore
export SD15=${SD15:-$HOME/models/sd15}       # local diffusers folder
export DATA=${DATA:-$HOME/datasets}          # holds pets/ and cub/
export FEATURES=${FEATURES:-$HOME/features}  # extraction output
export LORA_DIR=${LORA_DIR:-$HOME/lora}

export HF_HUB_OFFLINE=1                      # compute nodes have no network
source "$THESIS_ENV/bin/activate"
