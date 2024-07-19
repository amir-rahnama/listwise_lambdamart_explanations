# Local List-Wise Explanations of LambdaMART
This repository includes the code for the experiments for the paper, [Local List-Wise Explanations of LambdaMART](https://link.springer.com/chapter/10.1007/978-3-031-63797-1_19#Fn2) 
that is published. 

## TLDR;
In short, this repository includes the implementation of the following List-wise Explanation Techniques for explaining learning-to-rank models: 

* RankLIME
* GreedyScore
* PerMutation Importance (PMI)

And the aggregated point-wise explanations of: 
* LIRME, EXS
* LIME and KernelSHAP.

We evaluate explanations on the LambdaMART model trained on open LTR benchmarks: MQ2008, Web10K, and Yahoo.

### Our main finding: 
<p align="center">

<img width="621" alt="Screenshot 2024-07-19 at 11 17 23" src="https://github.com/user-attachments/assets/d1e1315b-98a0-4a81-a579-31210edd999f">
p>

* Explanations such as PMI and RankLIME can be faithful across datasets; however, only in a subset of the measures
* Overall, Greedy-Score, EXS (R), and LIME are not faithful across numerous measures and datasets.
* Among the (aggregated) point-wise explanations, LIRME explanations in all datasets and SHAP for the Web10k dataset provide relatively low fidelity ranks.

### Citation 
To cite the paper, please include the following Bibtex: 

```latex
@inproceedings{rahnama2024local,
  title={Local List-Wise Explanations of LambdaMART},
  author={Rahnama, Amir Hossein Akhavan and B{\"u}tepage, Judith and Bostr{\"o}m, Henrik},
  booktitle={World Conference on Explainable Artificial Intelligence},
  pages={369--392},
  year={2024},
  organization={Springer}
}
```



