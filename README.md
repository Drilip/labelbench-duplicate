## FlexMatch Fixed-Threshold Ablation
This branch contains the implementation used for the FlexMatch threshold ablation in our LabelBench reproducibility study.

The original FlexMatch implementation uses the class-dependent threshold 
`p_cutoff * beta_c / (2 - beta_c)`
where `beta_c` represents class-dependent learning progress. In this branch, the class-dependent thresholding rule is replaced by a constant confidence threshold of `0.95`. A pseudo-label is therefore accepted wheenver  `p_max >= 0.95`. All other FlexMatch training settings are unchanged. The modified trainer is implemented in:

`LabelBench/trainer/trainer_impl/flexmatch_fixed.py`

with configuration:

`configs/trainer/cifar10/flexmatch_fixed/clip_ViTB32_finetune.json`

Example:
python main.py \
  --seed 1234 \
  --wandb_name ATDL-labelbench \
  --dataset cifar10 \
  --data_dir ./data \
  --metric multi_class \
  --batch_size 1000 \
  --num_batch 10 \
  --embed_model_config none.json \
  --classifier_model_config clip_ViTB32_pretrained.json \
  --strategy_config random_sampling.json \
  --trainer_config cifar10/flexmatch_fixed/clip_ViTB32_finetune.json \
  --corrupter_config noiseless.json

## Results
The `results/semisl/` directory contains the extracted per-run trajectories and the aggregated mean/standard-error values used to produce the figures and values in the report.
