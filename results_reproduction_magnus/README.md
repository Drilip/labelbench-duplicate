# Fig. 1a reproduction 

CIFAR-10, CLIP ViT-B/32 + FlexMatch, batch size 1000, 10 rounds, 6 strategies x 4 seeds
(1234, 3334155, 6667077, 9999999). GALAXY and BAIT not run. Ablation: ../ablation_randinit_magnus/.

Run from this folder. The runs are already saved in cifar10/, so summarize.py and plot_training.py work right away. The two fetch commands
are only needed to download the runs from W&B again.

python fetch_mine.py --wandb_name ATDL-labelbench --dataset cifar10 --batch_size 1000 \
  --embed_model none --classifier_model clip_ViTB32_pretrained --trainer_config flexmatch

python fetch_mine.py --wandb_name ATDL-labelbench --dataset cifar10 --batch_size 1000 \
  --embed_model none --classifier_model clip_ViTB32_randinit --trainer_config flexmatch --group randinit

python summarize.py   # tables and t-tests

python plot_training.py --dataset cifar10 --batch_size 1000 --embed_model none \
  --classifier_model clip_ViTB32_pretrained --trainer_config flexmatch --metrics "Test Accuracy" \
  --strategies random confidence entropy margin coreset badge --smoothing max \
  --colors tab:blue tab:orange tab:green tab:red tab:purple tab:pink --y_lim .96 .982   # Fig. 1a plot

fetch_mine.py is a trimmed copy of the authors' script, with changes at the top of the file.
Needs wandb, numpy and matplotlib.
