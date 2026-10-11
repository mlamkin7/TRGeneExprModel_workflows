# TRGeneExprModel_workflows
Methods/Pipelines for generating figures for the paper "Incorporating tandem repeats improves gene expression prediction models"

## Figure Notebooks

Notebooks for generating main and supplementary figures found in the `figure_notebooks` folder. These include:

* `association_pval_validation.ipynb`: Creates main figures for the evaluation of linear and nonlinear tandem repeat association tests in Geuvadis and GTEx LCLs.
* `data_and_model_validation.ipynb`: Creates the majority main and supplementary figures surrounding the analysis of gene expression prediction models in SNP only, SNP+TR, and nonlinear settings. 
* `model_simulations.ipynb`: Creates main and supplementary figures about simulating causal SNPs and TRs with varying heritability. 
* `strat_plots.ipynb`: Creates main and supplementary figures about plotting nonlinear effects of tandem repeats. 

## Analysis Scripts

Main scripts for model training/imputing, eSTR effect detection, and main/supplementary figures found in the `scripts` folder. These include:

* `gene_model_pred.py`: Main analysis script that trains and outputs gene expression prediction models for SNP only, SNP+TR, and nonlinear settings.
* `impute_expression.py`:
* `nonlinear_str_gene_detection.py`:
* `expression_comparison.py`:
* `model_sims.py`:
* `ball_and_stick_plots.py`:
* `ld_gain_plots.py`:
