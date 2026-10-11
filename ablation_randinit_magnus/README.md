# Ablation: random-init CLIP visual encoder

Fig. 1(a) setup, but the CLIP visual encoder is initialized with random weights (Random and Confidence, 3 seeds).

- clip.py: copy over LabelBench/model/model_impl/clip.py (the only change to the authors' code)
- clip_ViTB32_randinit.json: copy into configs/model/
- launch_randinit.sh: starts the six runs

Without the modified clip.py, the config option is ignored and the runs use the pretrained encoder.

bash launch_randinit.sh 3 (run on UCloud, 2 GPUs, needs wandb login)

Tables and t-tests in the report: ../results_reproduction_magnus/summarize.py
