# TRGeneExprModel_workflows
Methods/Pipelines for generating figures for the paper "Incorporating tandem repeats improves gene expression prediction models".

## Figure Notebooks

Notebooks for generating main and supplementary figures found in the `figure_notebooks` folder. These include:

* `association_pval_validation.ipynb`: Creates main figures for the evaluation of linear and nonlinear tandem repeat association tests in Geuvadis and GTEx LCLs.
* `data_and_model_validation.ipynb`: Creates the majority main and supplementary figures surrounding the analysis of gene expression prediction models in SNP only, SNP+TR, and nonlinear settings. 
* `model_simulations.ipynb`: Creates main and supplementary figures about simulating causal SNPs and TRs with varying heritability. 
* `strat_plots.ipynb`: Creates main and supplementary figures about plotting nonlinear effects of tandem repeats. 

## Analysis Scripts

Main scripts for model training/imputing, eSTR effect detection, and main/supplementary figures found in the `scripts` folder. These include:

* `gene_model_pred.py`: Main analysis script that trains and outputs gene expression prediction models for SNP only, SNP+TR, and nonlinear settings.
* `impute_expression.py`: Script that takes in model, variant, and scaler outputs from `gene_model_pred.py` and applies them to new genotype data to impute expression. 
* `nonlinear_str_gene_detection.py`: Main script that takes in tandem repeat genotypes and gene expression data and runs linear and quadratic associations tests. 
* `expression_comparison.py`: Script that computes R-squared for GTEx imputed expression. 
* `model_sims.py`: Script that performs simulations that simulate causal variants and how they impact model accuracy for SNP only and SNP+TR models. 
* `ball_and_stick_plots.py`: Script that creates plots that compare the coefficients of the non-zero features present in the SNP only and SNP+TR models. Additionally, it outputs a bar plot for each TR genotype and the respective gene expression for EUR and YRI individuals.
* `ld_gain_plots.py`: Script that creates supplementary plots surrounding how LD impacts model R-squared gain. 
