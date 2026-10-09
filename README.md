# TRGeneExprModel_workflows
Methods/Pipelines for generating figures for the paper "Incorporating tandem repeats improves gene expression prediction models"

Files
1. association_pval_validation.ipynb - Evaluation of linear and nonlinear tandem repeat association tests in Geuvadis and GTEx LCLs.
2. ball_and_stick_plots.py - Script to generate plots that compare the features and their coefficients for the best SNP only and SNP+TR models. Also creates gene expression plots of the top tandem repeats for EUR and YRI data.
3. data_and_model_validation.ipynb - Majority of figures and analyses performed here. Covers gene expression model analysis within population, cross population, and cross dataset for linear and nonlinear models. 
4. expression_comparison.py - Output's predicted R-squared by comparing true expression to predicted expression in EUR and AFR samples in GTEx. 
5. gene_model_pred.py - Tool to create gene expression prediction models and use those models to make prediction on input data. 
6. impute_expression.py - Creates predicted expression from an input of genotypes, model, and model variants. 
7. ld_gain_plots.py - Script to create plots comparing LD and gain in EUR and YRI samples. 
8. model_sims.py - Simulation script for multi and single causal variable scenarios. 
9. strat_plots.ipynb - Creates plots that showcase the nonlinearity of TRs by holding an allele copy number constant and varying the second allele copy number, doing this for each unique allele copy number. 
