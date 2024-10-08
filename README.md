# 2021.0009

[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

The software and data in this repository are a snapshot of the software and data
that were used in the research reported on in the paper 
[Disjunctive Rule Sets](https://doi.org/10.1287/ijoc.2022.1242) 
by T. Wang and R. Ragodos. 

This is the code for Disjunctive Rule Set, which is a rule-based model for classification.

## Cite
To cite this software, please cite the [paper](https://doi.org/10.1287/ijoc.2022.1242) using its DOI and the software itself using the following DOI: [10.1287/ijoc.2022.1242.cd](https://doi.org/10.1287/ijoc.2022.1242.cd)

Below is the BibTex for citing this version of the code.

```
@article{DisRL2022,
author = {R. Ragodos and T. Wang},
publisher = {INFORMS Journal on Computing},
title = {Disjunctive Rule Lists},
year = {2022},
doi = {10.1287/ijoc.2022.1242.cd},
note = {https://github.com/INFORMSJoC/2021.0009}
}
```



# Results

Table 6 in the paper shows the performance of Disjunctive Rule Sets and various other rule-based  baselines. 
<img width="631" alt="Screen Shot 2022-08-02 at 12 28 11 AM" src="https://user-images.githubusercontent.com/3459074/182298648-0860305d-ea47-4543-8acc-ef45c028683a.png">


Figure 4 in the paper shows the trade-off between predictive performance and model size
<img width="654" alt="Screen Shot 2022-08-02 at 12 29 07 AM" src="https://user-images.githubusercontent.com/3459074/182298760-b3729b83-382a-42df-b9bd-0fdeff0b0a4c.png">


# Replicating

All public data used in the paper are in the folder "data" and the results of all models including the baselines in in the folder "results". The source code is in the folder "src". To replicate the experiments in the paper, run files in the folder "scripts"

To replicate the results in Table 6, run main.py and benchmark_synthetic.py in the scripts folder.

To generate Figure 4, run Result_Plotter.ipynb
