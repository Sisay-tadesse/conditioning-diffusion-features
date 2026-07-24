from pathlib import Path


ZFS       = Path("/zfsstore/user/s4184343")
SD15_DIR  = ZFS / "models" / "sd15"   # folder with model_index.json
DATA_DIR  = ZFS / "datasets"          # pets/ cub/ cifar/
FEAT_DIR  = ZFS / "features"     # cache: {dataset}/t{t}_{block}_{prompt}.pt
LORA_DIR  = ZFS / "lora"
REPO      = Path(__file__).resolve().parents[1]
RESULTS   = REPO / "results" / "results.csv"

# experiment grid (E1)
TIMESTEPS = [50, 100, 200, 300, 500]
BLOCKS = {   
    "down_2": "down_blocks.2",
    "mid":    "mid_block",
    "up_0":   "up_blocks.0",
    "up_1":   "up_blocks.1",
    "up_2":   "up_blocks.2",
}
DATASETS = ["pets", "cub", "cifar"]   # (cifar = CIFAR-100)
SEED = 0
