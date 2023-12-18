# Single-Pass Contrastive Learning Can Work for Both Homophilic and Heterophilic Graph (SPGCL)

[![arXiv](https://img.shields.io/badge/arXiv-2212.13350-b31b1b.svg)](https://arxiv.org/abs/2212.13350)
[![TMLR](https://img.shields.io/badge/OpenReview-8A2BE2.svg)](https://openreview.net/forum?id=244KePn09i&noteId=ea3TPPeLws)

This is the official repository of Single-Pass Graph Contrastive Learning (SPGCL), which is an agumentation-free GCL method introduced in our paper [Single-Pass Contrastive Learning Can Work for Both Homophilic and Heterophilic Graph](https://arxiv.org/abs/2211.10890).

The paper presents a Single-Pass Graph Contrastive Learning (SP-GCL) method that addresses the limitations of existing graph contrastive learning techniques, which typically require two forward passes and lack strong performance guarantees, especially on heterophilic graphs. By theoretically analyzing the concentration property of features obtained through neighborhood aggregation, the study introduces a single-pass, augmentation-free graph contrastive learning loss. This new approach is empirically validated on 14 benchmark datasets, demonstrating its ability to match or outperform existing methods with significantly lower computational overhead, making it applicable in real-world scenarios​​.





## :wrench: Prepare dataset
1. Download the dataset from [link](https://github.com/CUAI/Non-Homophily-Large-Scale).
2. Move the files under the `data` folder into the `dataset/non_homophilous_benchmark_data` folder.

## :keyboard: Usage
The examples are provided in `run.sh`.

``
bash run.sh
``


## :open_book: Why Single-Pass GCL?

### Single-Pass Design
SPGCL eliminates the need for two forward passes per instance, a common requirement in traditional graph contrastive learning, and Significantly reduces computational overhead compared to dual-pass methods.
![](https://github.com/haonan3/SPGCL/blob/main/results/SPGCL_pipeline.png?raw=true)

### Applicability to Heterophilic Graphs
Effectively works on both homophilic (where connected nodes have similar labels) and heterophilic graphs (where connected nodes typically have different labels). SPGCL expands the scope of graph contrastive learning to more diverse real-world scenarios.

### Performance Guarantees
Theoretical analysis offers strong performance guarantees for the minimizer of the loss on downstream tasks, which ensures reliability and predictability in various applications.

### Empirical Evidence 
From tests on 14 benchmark datasets, we show that SP-GCL can match or surpass existing methods in performance while reducing computational overhead, making it a more practical and efficient choice for real-world graph analysis tasks​​.
![](https://github.com/haonan3/SPGCL/blob/main/results/SPGCL_exp.png?raw=true)


## :paperclip:Citation

```
@article{
wang2023singlepass,
title={Single-Pass Contrastive Learning Can Work for Both Homophilic and Heterophilic Graph},
author={Haonan Wang and Jieyu Zhang and Qi Zhu and Wei Huang and Kenji Kawaguchi and Xiaokui Xiao},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2023},
url={https://openreview.net/forum?id=244KePn09i},
}
```
