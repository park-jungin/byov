# BYOV
[**SUBMISSION#17417 Bootstrap Your Own Views: Masked Ego-Exo Modeling for Fine-grained View-invariant Video Representations**]

## Installation
Build a conda environment from ``environment.yml``
```
conda env create --file environment.yml
conda activate byov
```

## Data Preparation
Download AE2 data and models [here](https://drive.google.com/drive/folders/1-v-5M5xTq8J7KDEgGQi2JhKRsQvjRaAo?usp=share_link) and save them to your designated data path.   
Modify `--dataset_root` in `utils/config.py` to be your data path.  
Note: avoid having ''ego'' in your root data path, as this could lead to potential issues.

## Evaluation
Evaluation of the learned representations on four downstream tasks:
+ (1) Action phase classification (regular, ego2exo and exo2ego)
+ (2) Frame retrieval (regular, ego2exo and exo2ego)
+ (3) Action phase progression
+ (4) Kendall's tau

## Training
We provide training scripts for BYOV on four datasets. Be sure to modify `--dataset_root` in `utils/config.py` to be your data path.
Training logs and checkpoints will be saved to `./logs/exp_{dataset_name}/{args.output_dir}`.

```shell
bash scripts/run.sh dataset_name  # choose among {break_eggs, pour_milk, pour_liquid, tennis_forehand}
```
