# Ablation Study

## Experiments

This branch focuses on four main variants:

- `Baseline`
- `Baseline + Containment`
- `Baseline + Ring Consistency`
- `Baseline + Ring Consistency + Containment` (`hierarchical_full`)

The contrastive branch is no longer treated as the main direction because earlier runs showed a consistent negative effect on both membrane-nucleus balance and overall robustness.

## Summary Table

### Membrane

| Method | BDQ | BSQ | BPQ | AJI |
|---|---:|---:|---:|---:|
| Baseline | 0.9199 | 0.7392 | 0.6805 | 0.6991 |
| Containment | 0.9210 | 0.7395 | 0.6817 | 0.7010 |
| Ring Only | 0.9161 | 0.7464 | 0.6844 | 0.7049 |
| Hierarchical Full | 0.9210 | 0.7440 | 0.6857 | 0.7021 |

### Nucleus

| Method | BDQ | BSQ | BPQ | AJI |
|---|---:|---:|---:|---:|
| Baseline | 0.8952 | 0.7268 | 0.6510 | 0.7001 |
| Containment | 0.8974 | 0.7243 | 0.6504 | 0.6980 |
| Ring Only | 0.8961 | 0.7251 | 0.6501 | 0.6994 |
| Hierarchical Full | 0.8988 | 0.7241 | 0.6511 | 0.6951 |

## Delta vs Baseline

### Membrane

| Method | Delta BDQ | Delta BSQ | Delta BPQ | Delta AJI |
|---|---:|---:|---:|---:|
| Containment | +0.0011 | +0.0004 | +0.0013 | +0.0020 |
| Ring Only | -0.0038 | +0.0073 | +0.0039 | +0.0058 |
| Hierarchical Full | +0.0011 | +0.0049 | +0.0053 | +0.0030 |

### Nucleus

| Method | Delta BDQ | Delta BSQ | Delta BPQ | Delta AJI |
|---|---:|---:|---:|---:|
| Containment | +0.0022 | -0.0025 | -0.0006 | -0.0022 |
| Ring Only | +0.0008 | -0.0016 | -0.0010 | -0.0007 |
| Hierarchical Full | +0.0036 | -0.0027 | +0.0001 | -0.0050 |

## Main Findings

### 1. Containment is helpful but asymmetric

Containment alone gives a small positive effect on membrane quality, especially in `AJI` and `BPQ`.
However, it slightly suppresses nucleus quality, which suggests that the spatial prior is improving structural plausibility more than dense nucleus segmentation quality.

### 2. Ring consistency is the main new gain source

`Ring Only` gives the strongest membrane improvement among the current ablations:

- best membrane `BSQ`
- best membrane `AJI`
- clear positive gain in membrane `BPQ`

This indicates that explicitly supervising the membrane-minus-nucleus region is more effective than the previous contrastive design for this dataset.

### 3. Hierarchical full is strong, but not the most balanced

`Hierarchical Full` gives:

- the best membrane `BPQ`
- the best nucleus `BDQ`
- roughly baseline-level nucleus `BPQ`

But it also causes the largest nucleus `AJI` drop among the new branch variants.
This means the joint use of ring consistency and containment is competitive, but currently a bit too strong on the nucleus side.

### 4. Current best practical choice

At the moment, the most stable main-method candidate is:

- `Ring Only`

Reason:

- it provides the clearest membrane gain
- it avoids the stronger nucleus penalty seen in `hierarchical_full`
- it is easier to justify as the primary innovation module

`Hierarchical Full` remains valuable as a supplementary structural variant and as evidence that combining priors can still improve some metrics, but it is not yet the cleanest final model.

## Paper-Oriented Interpretation

The current results support the following narrative:

- The previous feature-level contrastive route was not well matched to this small-batch paired-structure setting.
- Dense structural supervision is more effective than object-level contrastive alignment for this task.
- The hierarchical ring region acts as the main discriminative structural cue.
- The containment prior is still useful, but should be treated as an auxiliary topology prior rather than the dominant performance driver.

Suggested final framing:

- Core innovation:
  `Hierarchical Region Decomposition Consistency`
- Auxiliary innovation:
  `Membrane-Nucleus Containment Constraint`

Suggested full method name:

- `Structure-Prior Enhanced SAM2 with Hierarchical Region Consistency`

## Whether To Run A Tuned Version

Yes, one tuned run is still worth doing.

The current pattern suggests:

- `ring` is clearly useful
- `containment` is probably slightly too strong when combined with `ring`

So the most valuable next experiment is not a broad sweep, but one targeted tuned variant:

- `hierarchical_full_tune`

Recommended change:

- keep `loss_ring = 0.5`
- reduce `loss_contain` from `1.0` to `0.3`

Rationale:

- membrane gains are already strong
- the main issue is the nucleus `AJI` drop
- weakening containment has the highest chance of preserving structural benefit while reducing the nucleus penalty

## Recommended Next Command

```bash
python training/train.py --config configs/sam2.1_training/sam2.1_hiera_b+_trop2_hierarchical_priors.yaml --use-cluster 0 --num-gpus 1 --hydra-override ++trainer.loss.all.weight_dict.loss_contain=0.3 --hydra-override ++trainer.loss.all.weight_dict.loss_ring=0.5 --hydra-override ++launcher.experiment_log_dir=checkpoints/ablations/hierarchical_full_tune
```

Then evaluate with:

```bash
python infer.py --eval --mode test --ckpt-path checkpoints/ablations/hierarchical_full_tune/checkpoints/checkpoint.pt --experiment-tag hierarchical_full_tune --save-metrics
```

## Notes

- **BDQ**: detection-quality term from the PQ decomposition
- **BSQ**: segmentation-quality term from the PQ decomposition
- **BPQ**: panoptic quality
- **AJI**: aggregated Jaccard index
