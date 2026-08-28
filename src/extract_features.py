"""Pooled UNet activations from SD-1.5. One forward pass per image, no denoising.

    python extract_features.py --dataset pets --split trainval --t 100
    python extract_features.py --dataset cub  --split train    --t 300 --prompt P3 --cfg 5

Needs $SD15 (local diffusers folder) and $DATA.
"""
import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

BLOCKS = ["down_2", "mid", "up_0", "up_1", "up_2"]

# P3/P4 carry the true class name: oracle, not deployable.
PROMPTS = {
    "pets": {
        "P0": "",
        "P1": "a photo",
        "P2": "a photo of a pet",
        "P3": "a photo of a {c}",
        "P4": "a high-quality photograph of a {c}, a type of pet",
    },
    "cub": {
        "P0": "",
        "P1": "a photo",
        "P2": "a photo of a bird",
        "P3": "a photo of a {c}",
        "P4": "a high-quality photograph of a {c}, a type of bird",
    },
}

TRANSFORM = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),  # [-1, 1]
])


class Pets(Dataset):
    """Pets from images/ + annotations/, bypassing torchvision's layout rules."""

    def __init__(self, root, split="trainval", transform=None):
        self.transform = transform or TRANSFORM
        root = Path(root)
        if not (root / "annotations").exists():
            root = root / "oxford-iiit-pet"  # torchvision nests it
        self.samples, names = [], {}
        with open(root / "annotations" / f"{split}.txt") as f:
            for line in f:
                name, cls = line.split()[:2]
                label = int(cls) - 1
                self.samples.append((root / "images" / f"{name}.jpg", label))
                names[label] = name.rsplit("_", 1)[0].replace("_", " ")
        self.classes = [names[i] for i in range(len(names))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("RGB")), label


class CUB(Dataset):
    """CUB from the extracted tarball."""

    def __init__(self, root, split="train", transform=None):
        self.transform = transform or TRANSFORM
        root = Path(root)
        img_dir = root / "images"
        ids = {}
        with open(root / "images.txt") as f:
            for line in f:
                i, p = line.split()
                ids[i] = p
        labels = {}
        with open(root / "image_class_labels.txt") as f:
            for line in f:
                i, c = line.split()
                labels[i] = int(c) - 1
        is_train = {}
        with open(root / "train_test_split.txt") as f:
            for line in f:
                i, t = line.split()
                is_train[i] = t == "1"
        want = split == "train"
        self.samples = [(img_dir / ids[i], labels[i])
                        for i in ids if is_train[i] == want]
        with open(root / "classes.txt") as f:
            # "001.Black_footed_Albatross"
            self.classes = [line.split()[1].split(".", 1)[1].replace("_", " ")
                            for line in f]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("RGB")), label


def get_dataset(name, split, data_root, transform=None):
    if name == "pets":
        ds = Pets(os.path.join(data_root, "pets"), split=split,
                  transform=transform)
        return ds, ds.classes
    if name == "cub":
        ds = CUB(os.path.join(data_root, "cub", "CUB_200_2011"), split=split,
                 transform=transform)
        return ds, ds.classes
    raise ValueError(name)


class SDFeatureExtractor:
    def __init__(self, sd_path, device="cuda", dtype=torch.float16, lora=None):
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            sd_path, torch_dtype=dtype, safety_checker=None)
        if lora:
            # offline: the filename has to be explicit
            pipe.load_lora_weights(
                lora, weight_name="pytorch_lora_weights.safetensors")
            pipe.fuse_lora()  # so the plain forward sees it
        pipe.to(device)
        self.vae = pipe.vae
        self.unet = pipe.unet
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.scheduler = pipe.scheduler  # add_noise() only
        self.device, self.dtype = device, dtype

        self._feats = {}
        modules = {
            "down_2": self.unet.down_blocks[2],
            "mid":    self.unet.mid_block,
            "up_0":   self.unet.up_blocks[0],
            "up_1":   self.unet.up_blocks[1],
            "up_2":   self.unet.up_blocks[2],
        }
        for name, mod in modules.items():
            mod.register_forward_hook(self._make_hook(name))

    def _make_hook(self, name):
        def hook(module, inputs, output):
            # down blocks return a tuple, mid and up a tensor
            x = output[0] if isinstance(output, tuple) else output
            self._feats[name] = x.detach()
        return hook

    @torch.no_grad()
    def _encode_prompts(self, prompts):
        """One prompt per image, not per batch: P3/P4 differ per row."""
        tok = self.tokenizer(prompts, padding="max_length",
                             max_length=self.tokenizer.model_max_length,
                             truncation=True, return_tensors="pt")
        return self.text_encoder(tok.input_ids.to(self.device))[0]

    @torch.no_grad()
    def _unet_pass(self, noisy, ts, prompts):
        emb = self._encode_prompts(prompts)
        self._feats.clear()
        self.unet(noisy, ts, encoder_hidden_states=emb)
        # fp32 for the mean; fp16 sums over 4096 positions drift
        return {k: v.float().mean(dim=(2, 3)).cpu() for k, v in self._feats.items()}

    @torch.no_grad()
    def extract(self, images, t, prompts, cfg_scale=None, seed=0):
        """images: [B,3,512,512] in [-1,1], one prompt per image.

        With cfg_scale, also runs the empty prompt and mixes the two pooled
        activations. Both branches come back; cond is the control.
        """
        images = images.to(self.device, self.dtype)
        latents = self.vae.encode(images).latent_dist.mode() \
            * self.vae.config.scaling_factor      # mode(), not sample()
        # same noise for every config, so only the conditioning varies
        g = torch.Generator(self.device).manual_seed(seed)
        noise = torch.randn(latents.shape, generator=g,
                            device=self.device, dtype=self.dtype)
        ts = torch.full((latents.shape[0],), t, device=self.device, dtype=torch.long)
        noisy = self.scheduler.add_noise(latents, noise, ts)

        cond = self._unet_pass(noisy, ts, prompts)
        if cfg_scale is None:
            return cond
        uncond = self._unet_pass(noisy, ts, [""] * len(prompts))
        cfg = {k: uncond[k] + cfg_scale * (cond[k] - uncond[k]) for k in cond}
        return {"cond": cond, "cfg": cfg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["pets", "cub"])
    ap.add_argument("--split", required=True,
                    help="pets: trainval|test; cub: train|test")
    ap.add_argument("--t", type=int, required=True, help="noise level in [0,1000)")
    ap.add_argument("--prompt", default="P0", choices=list(PROMPTS["pets"]))
    ap.add_argument("--cfg", type=float, default=None,
                    help="guidance scale; omit for a single pass")
    ap.add_argument("--lora", default=None, help="trained adapter folder")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N images")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sd", default=os.environ.get("SD15"),
                    help="local SD-1.5 diffusers folder (default: $SD15)")
    ap.add_argument("--data", default=os.environ.get("DATA",
                    os.path.expanduser("~/datasets")))
    ap.add_argument("--out", default=os.path.expanduser("~/features"))
    args = ap.parse_args()
    assert args.sd, "set $SD15 or pass --sd"

    ds, classes = get_dataset(args.dataset, args.split, args.data)
    if args.limit:
        ds = torch.utils.data.Subset(ds, range(min(args.limit, len(ds))))
    # unshuffled: rows have to keep matching labels
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    ex = SDFeatureExtractor(args.sd, lora=args.lora)
    template = PROMPTS[args.dataset][args.prompt]

    acc, labels_all = {}, []
    for i, (images, labels) in enumerate(loader):
        prompts = [template.format(c=classes[y]) for y in labels]
        out = ex.extract(images, args.t, prompts, cfg_scale=args.cfg,
                         seed=args.seed)
        flat = ({f"cond/{k}": v for k, v in out["cond"].items()}
                | {f"cfg/{k}": v for k, v in out["cfg"].items()}
                ) if args.cfg is not None else out
        for k, v in flat.items():
            acc.setdefault(k, []).append(v)
        labels_all.append(labels)
        print(f"batch {i + 1}/{len(loader)}", flush=True)

    feats = {k: torch.cat(v) for k, v in acc.items()}
    labels = torch.cat(labels_all)

    tag = (f"{args.dataset}_{args.split}_t{args.t}_{args.prompt}"
           f"_cfg{args.cfg if args.cfg is not None else 'off'}"
           f"{'_lora' if args.lora else ''}"
           f"{'_limit' + str(args.limit) if args.limit else ''}")
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, tag + ".pt")
    torch.save({"feats": feats, "labels": labels, "classes": classes,
                "meta": vars(args)}, path)

    n = labels.shape[0]
    print(f"saved {path}")
    for k, v in feats.items():
        print(f"  {k}: {tuple(v.shape)}")
    print(f"done: {n} images")


if __name__ == "__main__":
    main()
