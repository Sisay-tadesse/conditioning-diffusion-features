"""DINOv2, CLIP and ResNet-50 features, shaped so probe.py cannot tell them from
the diffusion ones.

Prime the weight caches on a login node first; compute nodes have no network.

    python src/extract_baselines.py --prime
    python src/extract_baselines.py
"""
import argparse
import os

# on zfsstore: visible from every node, off the home quota.
# setdefault so a job script can still override.
os.environ.setdefault("HF_HOME", "/zfsstore/user/s4184343/hf-cache")
os.environ.setdefault("TORCH_HOME", "/zfsstore/user/s4184343/torch-cache")

import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms

from extract_features import get_dataset

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
CLIPSTAT = ((0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711))

SPLITS = {"pets": ("trainval", "test"), "cub": ("train", "test")}
MODELS = ("dinov2", "clip", "resnet50")

# no safetensors on main, and the .bin wants torch>=2.6. --prime converts it
# once, online; everything after that loads this folder.
CLIP_DIR = os.environ.get("CLIP_DIR",
                          "/zfsstore/user/s4184343/models/clip-vitb16-vision")


def make_tf(resize, crop, stats):
    return transforms.Compose([
        transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])


def build(name, device):
    """-> (images -> vectors, the transform that model expects)."""
    if name == "dinov2":
        from transformers import Dinov2Model
        m = Dinov2Model.from_pretrained("facebook/dinov2-base").to(device).eval()
        return (lambda x: m(pixel_values=x).pooler_output,   # CLS token, 768-d
                make_tf(256, 224, IMAGENET))                 # DINOv2 eval recipe
    if name == "clip":
        from transformers import CLIPVisionModelWithProjection
        if not os.path.isdir(CLIP_DIR):  # first run, online only
            CLIPVisionModelWithProjection.from_pretrained(
                "openai/clip-vit-base-patch16",
                use_safetensors=True).save_pretrained(CLIP_DIR)
        m = CLIPVisionModelWithProjection.from_pretrained(
            CLIP_DIR).to(device).eval()
        return (lambda x: m(pixel_values=x).image_embeds,   # projected, 512-d
                make_tf(224, 224, CLIPSTAT))                # CLIP eval recipe
    if name == "resnet50":
        w = models.ResNet50_Weights.IMAGENET1K_V2
        m = models.resnet50(weights=w)
        m.fc = torch.nn.Identity()                          # penultimate, 2048-d
        m = m.to(device).eval()
        return m, w.transforms()  # torchvision ships the V2 recipe
    raise ValueError(name)


@torch.no_grad()
def extract(model_name, dataset, split, args, device):
    fn, tf = build(model_name, device)
    ds, classes = get_dataset(dataset, split, args.data, transform=tf)
    if args.limit:
        ds = torch.utils.data.Subset(ds, range(min(args.limit, len(ds))))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    feats, labels_all = [], []
    for i, (images, labels) in enumerate(loader):
        feats.append(fn(images.to(device)).float().cpu())
        labels_all.append(labels)
        if (i + 1) % 20 == 0:
            print(f"  batch {i + 1}/{len(loader)}", flush=True)

    tag = (f"{dataset}_{split}_{model_name}"
           f"{'_limit' + str(args.limit) if args.limit else ''}")
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, tag + ".pt")
    torch.save({"feats": {model_name: torch.cat(feats)},
                "labels": torch.cat(labels_all), "classes": classes,
                "meta": {"dataset": dataset, "t": "-", "prompt": "-",
                         "cfg": None, "lora": None, "model": model_name}},
               path)
    print(f"saved {path} ({len(torch.cat(labels_all))} images)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, default=None, help="default: all")
    ap.add_argument("--dataset", choices=list(SPLITS), default=None,
                    help="default: both")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--data", default=os.environ.get("DATA",
                    os.path.expanduser("~/datasets")))
    ap.add_argument("--out", default=os.path.expanduser("~/features"))
    ap.add_argument("--prime", action="store_true",
                    help="fill the caches and exit (login node)")
    args = ap.parse_args()

    if args.prime:
        for m in MODELS:
            print(f"priming {m} ...")
            build(m, "cpu")
        print("caches primed")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    for m in ([args.model] if args.model else MODELS):
        for ds in ([args.dataset] if args.dataset else SPLITS):
            for split in SPLITS[ds]:
                print(f"=== {m} / {ds} / {split}")
                extract(m, ds, split, args, device)
    print("all done")


if __name__ == "__main__":
    main()
