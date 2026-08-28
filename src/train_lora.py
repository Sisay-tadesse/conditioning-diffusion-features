"""LoRA on the SD-1.5 UNet, trained with the plain denoising objective.

No classification signal anywhere: the class name enters only as prompt text.
Whatever the probe finds afterwards is a side effect. Attention projections
only, so the trunk, VAE and text encoder stay frozen.

    python src/train_lora.py --dataset pets --data $SCRATCH/data --out ~/lora/pets_r16
"""
import argparse
import math
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# same loaders and transform as the extractor, so both see identical pixels
from extract_features import get_dataset

CKPT_STEPS = {1000, 2500, 5000}
PROMPT = "a photo of a {c}"


def save_ckpt(pipe_cls, unet, out, step):
    from peft.utils import get_peft_model_state_dict
    d = os.path.join(out, f"ckpt-{step}")
    os.makedirs(d, exist_ok=True)
    pipe_cls.save_lora_weights(
        d, unet_lora_layers=get_peft_model_state_dict(unet),
        safe_serialization=True)
    print(f"[ckpt] saved {d}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["pets", "cub"])
    ap.add_argument("--data", default=os.environ.get("DATA", os.path.expanduser("~/datasets")))
    ap.add_argument("--out", required=True)
    ap.add_argument("--sd", default=os.environ.get("SD15"))
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2,
                    help="effective batch = batch_size * grad_accum")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert args.sd, "set $SD15 or pass --sd"
    torch.manual_seed(args.seed)
    device = "cuda"

    from diffusers import StableDiffusionPipeline, DDPMScheduler
    from peft import LoraConfig

    # fp32 weights, fp16 only inside autocast
    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd, torch_dtype=torch.float32, safety_checker=None)
    noise_sched = DDPMScheduler.from_config(pipe.scheduler.config)
    vae, text_encoder, unet, tok = pipe.vae, pipe.text_encoder, pipe.unet, pipe.tokenizer
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    unet.add_adapter(LoraConfig(
        r=args.rank, lora_alpha=args.rank,          # alpha = rank -> scale 1.0
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    unet.enable_gradient_checkpointing()           
    params = [p for p in unet.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in params)
    print(f"trainable params: {n_tr/1e6:.2f}M (rank {args.rank})", flush=True)

    vae.to(device); text_encoder.to(device); unet.to(device)
    opt = torch.optim.AdamW(params, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")

    # train split only; test unseen
    split = "trainval" if args.dataset == "pets" else "train"
    ds, classes = get_dataset(args.dataset, split, args.data)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)
    epochs = math.ceil(args.steps * args.grad_accum / len(loader))
    print(f"{len(ds)} imgs, {len(loader)} micro-batches/ep (bs {args.batch_size} "
          f"x accum {args.grad_accum}), ~{epochs} epochs for {args.steps} opt steps",
          flush=True)

    step, micro = 0, 0                 # step = optimiser steps, not batches
    unet.train()
    opt.zero_grad()
    for _ in range(epochs):
        for images, labels in loader:
            if step >= args.steps:
                break
            images = images.to(device)
            prompts = [PROMPT.format(c=classes[y]) for y in labels]
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample() \
                    * vae.config.scaling_factor
                ids = tok(prompts, padding="max_length", truncation=True,
                          max_length=tok.model_max_length,
                          return_tensors="pt").input_ids.to(device)
                emb = text_encoder(ids)[0]
            noise = torch.randn_like(latents)
            ts = torch.randint(0, noise_sched.config.num_train_timesteps,
                               (latents.shape[0],), device=device)
            noisy = noise_sched.add_noise(latents, noise, ts)

            with torch.autocast("cuda", dtype=torch.float16):
                pred = unet(noisy, ts, encoder_hidden_states=emb).sample
                loss = F.mse_loss(pred.float(), noise.float())
            scaler.scale(loss / args.grad_accum).backward()
            micro += 1
            if micro % args.grad_accum:
                continue

            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

            step += 1
            if step % 100 == 0:
                print(f"step {step}/{args.steps} loss {loss.item():.4f}", flush=True)
            if step in CKPT_STEPS:
                save_ckpt(StableDiffusionPipeline, unet, args.out, step)
        if step >= args.steps:
            break

    if args.steps not in CKPT_STEPS:
        save_ckpt(StableDiffusionPipeline, unet, args.out, step)
    print("done", flush=True)


if __name__ == "__main__":
    main()
