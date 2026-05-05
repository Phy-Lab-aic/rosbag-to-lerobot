"""Minimal ACT training smoke test against the aic_e2e_test dataset.

Loads the converted LeRobot v3 dataset, builds an ACTPolicy from the
dataset's metadata, and runs a handful of optimizer steps to confirm:
  * dataset + dataloader produce model-compatible batches
  * ACT forward/backward + optimizer step do not error
  * loss decreases (very loosely) over the smoke window

Not a real training run — just an end-to-end wiring check.
"""

import argparse
import time

import torch
from torch.utils.data import DataLoader

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy


def _build_features_from_dataset(ds: LeRobotDataset):
    """Map dataset features -> PolicyFeature entries ACT consumes."""
    input_features = {}
    output_features = {}
    for key, info in ds.features.items():
        shape = tuple(info["shape"])
        if key.startswith("observation.images."):
            # ACT expects (C, H, W). Dataset stores (H, W, C); torchcodec
            # already returns (C, H, W) at sample time, so use that shape.
            h, w, c = shape
            input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(c, h, w))
        elif key == "observation.state":
            input_features[key] = PolicyFeature(type=FeatureType.STATE, shape=shape)
        elif key == "action":
            output_features[key] = PolicyFeature(type=FeatureType.ACTION, shape=shape)
    return input_features, output_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/weed/aic_results/lerobot_output_e2e_test/aic_e2e_test")
    parser.add_argument("--repo-id", default="aic_e2e_test")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"[load] LeRobotDataset(root={args.root})")
    ds = LeRobotDataset(repo_id=args.repo_id, root=args.root)
    print(f"  num_episodes={ds.num_episodes}  num_frames={ds.num_frames}  fps={ds.fps}")

    # ACT consumes a temporal action chunk; LeRobotDataset returns a chunk
    # when delta_indices is set. Re-instantiate with action delta indices.
    delta_timestamps = {"action": [i / ds.fps for i in range(args.chunk_size)]}
    ds = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.root,
        delta_timestamps=delta_timestamps,
    )

    input_features, output_features = _build_features_from_dataset(ds)
    print(f"[features] inputs={list(input_features)}  outputs={list(output_features)}")

    cfg = ACTConfig(
        n_obs_steps=1,
        chunk_size=args.chunk_size,
        n_action_steps=args.chunk_size,
        input_features=input_features,
        output_features=output_features,
        normalization_mapping={
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        },
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        dim_model=256,
        n_heads=4,
        dim_feedforward=512,
        n_encoder_layers=2,
        n_decoder_layers=1,
        use_vae=False,
    )

    print("[build] ACTPolicy(...)")
    policy = ACTPolicy(cfg, dataset_stats=ds.meta.stats).to(args.device)
    policy.train()

    optim = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(args.device == "cuda"),
        drop_last=True,
    )
    print(f"[loader] batches/epoch={len(loader)}  batch_size={args.batch_size}")

    losses = []
    t0 = time.time()
    step = 0
    while step < args.steps:
        for batch in loader:
            batch = {
                k: (v.to(args.device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            loss, _ = policy.forward(batch)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            losses.append(float(loss.detach().cpu()))
            step += 1
            print(f"  step {step:3d}/{args.steps}  loss={losses[-1]:.4f}")
            if step >= args.steps:
                break

    dt = time.time() - t0
    print(f"\n[done] {args.steps} steps in {dt:.1f}s "
          f"({args.steps / dt:.2f} steps/s on {args.device})")
    print(f"[done] first_loss={losses[0]:.4f}  last_loss={losses[-1]:.4f}")


if __name__ == "__main__":
    main()
