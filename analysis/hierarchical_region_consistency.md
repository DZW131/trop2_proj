# Hierarchical Region Consistency

## Idea

This variant replaces the contrastive module with a denser structural objective:

- `loss_ring`: supervises the membrane-minus-nucleus ring region.
- `loss_contain`: preserves the membrane-contains-nucleus prior.

The ring target is defined from the paired annotations:

- `Y_ring = Y_membrane - Y_nucleus`

The predicted ring probability is computed from the two decoder outputs:

- `P_ring = sigmoid(P_membrane) * (1 - sigmoid(P_nucleus))`

The training loss adds a BCE term and a Dice term on the ring region.

## Recommended Experiments

Baseline:

```bash
python training/train.py --config configs/sam2.1_training/sam2.1_hiera_b+_trop2_hierarchical_priors.yaml --ablation baseline --use-cluster 0 --num-gpus 1
```

Ring only:

```bash
python training/train.py --config configs/sam2.1_training/sam2.1_hiera_b+_trop2_hierarchical_priors.yaml --ablation ring --use-cluster 0 --num-gpus 1
```

Containment plus ring:

```bash
python training/train.py --config configs/sam2.1_training/sam2.1_hiera_b+_trop2_hierarchical_priors.yaml --ablation hierarchical_full --use-cluster 0 --num-gpus 1
```

## Default Weights

- `loss_contain = 1.0`
- `loss_ring = 0.5`
- `loss_struct_contrast = 0.0`

## Paper Framing

Suggested module names:

- `Hierarchical Region Decomposition Consistency`
- `Membrane-Nucleus Containment Constraint`

Suggested full method name:

- `Structure-Prior Enhanced SAM2 with Hierarchical Region Consistency`
