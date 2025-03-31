# BYOV (CVPR 2025): Official Project Page
This repository provides the official PyTorch implementation of the following paper:
> Bootstrap Your Own Views: Masked Ego-Exo Modeling for Fine-grained View-invariant Video Representations<br>
> [Jungin Park](https://park-jungin.github.io/) ([Yonsei Univ.](https://www.yonsei.ac.kr)), [Jiyoung Lee](https://lee-jiyoung.github.io/)* ([Ewha Womans University](https://myr.ewha.ac.kr/deptai/index.do)), [Kwanghoon Sohn](https://diml.yonsei.ac.kr/students/)* (Yonsei Univ.) (*: corresponding authors)<br>
> Accepted to CVPR 2025<br>

> Paper: [arxiv](https://arxiv.org/abs/2503.19706)<br>

> **Abstract:** 
*View-invariant representation learning from egocentric (first-person, ego) and exocentric (third-person, exo) videos is a promising approach toward generalizing video understanding systems across multiple viewpoints. However, this area has been underexplored due to the substantial differences in perspective, motion patterns, and context between ego and exo views. In this paper, we propose a novel masked ego-exo modeling that promotes both causal temporal dynamics and cross-view alignment, called Bootstrap Your Own Views (BYOV), for fine-grained view-invariant video representation learning from unpaired ego-exo videos. We highlight the importance of capturing the compositional nature of human actions as a basis for robust cross-view understanding. Specifically, self-view masking and cross-view masking predictions are designed to learn view-invariant and powerful representations concurrently. Experimental results demonstrate that our BYOV significantly surpasses existing approaches with notable gains across all metrics in four downstream ego-exo video tasks. The code is available at https://github.com/park-jungin/byov.*<br>

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

## Citations
```
@inproceedings{park2025bootstrap,
  title={Bootstrap Your Own Views: Masked Ego-Exo Modeling for Fine-grained View-invariant Video Representations},
  author={Park, Jungin and Lee, Jiyoung and Sohn, Kwanghoon},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

## Acknowledgments
Our implementation is heavily derived from [AE2](https://github.com/zihuixue/AlignEgoExo).
Thanks to the AE2 implementation.
