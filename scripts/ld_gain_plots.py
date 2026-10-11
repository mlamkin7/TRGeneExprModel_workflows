import json
import pickle
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from scipy import stats
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from haptools.data import GenotypesTR, GenotypesVCF
from gene_model_pred import GeneExpressionPredictor

# NOTE Important so when loading figures into illustrator the text is seen as text
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42 


def plot_LD_diff(x_data, y_data, x_label, y_label, output):
    # Calculate pearson correlation for each plot
    result = stats.pearsonr(x_data, y_data)
    print(f"pearson R: {result.statistic} Pval: {result.pvalue}")

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.scatter(x_data, y_data)
    fig.savefig(output)
    fig.show()

    return


def main():
    # Files for loading GTs and expr
    #path = "/expanse/projects/gymreklab/mlamkin/" # Use for Expanse
    path = "/gymreklab-tscc/mlamkin/" # Use for Snorlax
    EUR_header_file = path + "GeneExprPredictor_data/train_data/EUR_normalized_and_filtered_hg38.csv"
    EUR_expr_file = path + "GeneExprPredictor_data/train_data/EUR_normalized_filtered_final_residuals_hg38.csv"
    AFR_header_file = path + "GeneExprPredictor_data/train_data/AFR_normalized_and_filtered_hg38.csv"
    AFR_expr_file = path + "GeneExprPredictor_data/train_data/AFR_normalized_filtered_final_residuals_hg38.csv"
    snp_file = path + "GeneExprPredictor_data/SNP_vcfs/SNP_{}_filtered.vcf.gz"
    str_file = path + "GeneExprPredictor_data/STR_vcfs/ensemble_{}_filtered.vcf.gz"

    """
    # Get list of 2553 nonzero genes 
    nonzero_gene_ids_file = path + "GeneExprPredictor_data/gene_sets/gtex/geuvadis_EUR_trained_genes.txt"
    gene_ids = np.loadtxt(nonzero_gene_ids_file, dtype=object)

    # Collect 
    gain_df = pd.read_csv(
        '/storage/mlamkin/data/GeneExprPredictor/paper_figures/trained_genes_table.csv', 
        delimiter=',', 
        usecols=['gene_name','gene','Num_SNPs','Num_TRs','adj_EUR_gain','AFR_gain']
    )

    # Results used to plot [gene_id, LD measure for EUR TR and SNP, LD for AFR TR and SNP, EUR gain, and AFR - EUR gain]
    plot_data = []

    for gene_id in gene_ids:
        # Grab gene gain and num_snp info
        gain_gene_df = gain_df[gain_df['gene'] == gene_id]
        num_snps = gain_gene_df['Num_SNPs'].values[0]

        # Load SNP+TR Model Coefficients
        snp_tr_model_file = f"/gymreklab-tscc/mlamkin/GeneExprPredictor_data/model_results/saved_models/geuvadis/{gene_id}_SNPs+STRs.pkl"
        with open(snp_tr_model_file, 'rb') as f:
            snp_tr_model = pickle.load(f)

        snp_tr_coeff = snp_tr_model.coef_
        tr_coeffs = np.abs(snp_tr_coeff[num_snps:])
        best_tr_ind = np.argmax(tr_coeffs)

        # If no TRs are used in the model don't put in plot
        if tr_coeffs[best_tr_ind] < 1e-8:
            continue

        # Load EUR and AFR data
        sample_ind = 5 # NOTE for GTEx sample_ind is 4
        gene_id_ind = 3
        gene_chrom_ind = 0
        gene_start_ind = 1
        gene_end_ind = 2

        eur_gene_expr_predictor = GeneExpressionPredictor(
            paramsearchfxn=None,
            model=None,
            model_params=None,
            hyperparams=None,
            gene_names=[gene_id],
            sample_ind=sample_ind,
            gene_id_ind=gene_id_ind,
            gene_chrom_ind=gene_chrom_ind,
            gene_start_ind=gene_start_ind,
            gene_end_ind=gene_end_ind,
            train_header_file=EUR_header_file,
            predict_header_file=None,
            snp_file=snp_file,
            str_file=str_file,
            train_expr_file=EUR_expr_file,
            predict_expr_file=None,
            param_shift=None,
            coeff_thresh=None,
            outpath=None,
            n_jobs=1,
            level="WARNING",
        )

        afr_gene_expr_predictor = GeneExpressionPredictor(
            paramsearchfxn=None,
            model=None,
            model_params=None,
            hyperparams=None,
            gene_names=[gene_id],
            sample_ind=sample_ind,
            gene_id_ind=gene_id_ind,
            gene_chrom_ind=gene_chrom_ind,
            gene_start_ind=gene_start_ind,
            gene_end_ind=gene_end_ind,
            train_header_file=AFR_header_file,
            predict_header_file=None,
            snp_file=snp_file,
            str_file=str_file,
            train_expr_file=AFR_expr_file,
            predict_expr_file=None,
            param_shift=None,
            coeff_thresh=None,
            outpath=None,
            n_jobs=1,
            level="WARNING",
        )

        eur_samples, header_genes = eur_gene_expr_predictor.prepare_header_info(eur_gene_expr_predictor.train_header_file)

        # Collect the variants
        eur_dataset, eur_expr = eur_gene_expr_predictor.collect_gts_gene_expr(
            gene_id,
            500,
            header_genes,
            eur_samples,
            snp_variants=None,
            str_variants=None,
            expr_file=eur_gene_expr_predictor.train_expr_file,
            header_file=eur_gene_expr_predictor.train_header_file,
        )

        eur_gts = eur_dataset[1]
        eur_gt_scaler = StandardScaler()
        norm_eur_gts = eur_gt_scaler.fit_transform(eur_gts)

        eur_variants = {}
        eur_vars = eur_gene_expr_predictor.variants[gene_id][:eur_dataset[1].shape[1]]
        for var_ind, var in enumerate(eur_vars):
            id = f"{var[0]}_{var[1]}_{var[2]}"
            eur_variants[id] = var_ind

        afr_samples, afr_header_genes = afr_gene_expr_predictor.prepare_header_info(afr_gene_expr_predictor.train_header_file)
        afr_dataset, afr_expr = afr_gene_expr_predictor.collect_gts_gene_expr(
            gene_id,
            500,
            afr_header_genes,
            afr_samples,
            snp_variants=eur_gene_expr_predictor.snp_variants[gene_id],
            str_variants=eur_gene_expr_predictor.str_variants[gene_id],
            expr_file=afr_gene_expr_predictor.train_expr_file,
            header_file=afr_gene_expr_predictor.train_header_file,
            filter_freq=False,
        )

        afr_gts = np.zeros((afr_dataset[1].shape[0], eur_dataset[1].shape[1]))
        afr_variants = afr_gene_expr_predictor.variants[gene_id][:afr_dataset[1].shape[1]]
        for afr_ind, afr_var in enumerate(afr_variants):
            id = f"{afr_var[0]}_{afr_var[1]}_{afr_var[2]}"
            afr_gts[:,eur_variants[id]] = afr_dataset[1][:,afr_ind]

        afr_gt_scaler = StandardScaler()
        norm_afr_gts = afr_gt_scaler.fit_transform(afr_gts)

        # Calculate LD between TRs and SNPs and grab max LD for best TR for EUR
        eur_ld_matrix = np.corrcoef(np.transpose(norm_eur_gts))**2
        eur_ld_matrix[np.isnan(eur_ld_matrix)] = 0
        eur_tr_correlations = eur_ld_matrix[num_snps:,:num_snps]
        eur_tr_max_ld_vals = np.max(eur_tr_correlations, axis=1)
        eur_tr_best_snp = np.argmax(eur_tr_correlations, axis=1)[best_tr_ind]
        best_tr_eur_ld = eur_tr_max_ld_vals[best_tr_ind]

        # Calculcate LD between TRs and SNPs and grab max LD for best TR for AFR
        afr_ld_matrix = np.corrcoef(np.transpose(norm_afr_gts))**2
        afr_ld_matrix[np.isnan(afr_ld_matrix)] = 0
        afr_tr_correlations = afr_ld_matrix[num_snps:,:num_snps]
        best_tr_afr_ld = afr_tr_correlations[best_tr_ind,eur_tr_best_snp]

        plot_data.append(
            [
                gene_id, 
                best_tr_eur_ld, 
                best_tr_afr_ld, 
                gain_gene_df['adj_EUR_gain'].values[0],
                gain_gene_df['AFR_gain'].values[0],
            ]
        )
    
    #plot_data_df = pd.DataFrame(plot_data, columns=['gene_id', 'EUR_TR_SNP_LD', 'AFR_TR_SNP_LD', 'adj_EUR_gain', 'AFR_gain'])
    #plot_data_df.to_csv('/storage/mlamkin/data/GeneExprPredictor/paper_figures/LD_gain_supp_plot_data.csv', index=False)
    """
    
    plot_data_df = pd.read_csv('/storage/mlamkin/data/GeneExprPredictor/paper_figures/LD_gain_supp_plot_data.csv')
    subset_plot_data_df = plot_data_df[plot_data_df['adj_EUR_gain'] > 0]
    print(plot_data_df.shape)
    print(subset_plot_data_df.shape)

    # Plot 1. Difference in Geuvadis EUR gain vs max LD in EUR
    plot_LD_diff(
        subset_plot_data_df['adj_EUR_gain'], 
        subset_plot_data_df['EUR_TR_SNP_LD'], 
        'European Gain (SNP+TR - SNP adjusted R-squared)', 
        'LD Correlation', 
        '/storage/mlamkin/data/GeneExprPredictor/paper_figures/EUR_LD_gain_supp.pdf'
    )

    # Plot 2. Difference in AFR - EUR gain vs max LD in EUR - max LD in AFR 
    plot_LD_diff(
        plot_data_df['AFR_gain'] - plot_data_df['adj_EUR_gain'], 
        plot_data_df['EUR_TR_SNP_LD'] - plot_data_df['AFR_TR_SNP_LD'], 
        'YRI - European Gain', 
        'European - YRI LD correlation', 
        '/storage/mlamkin/data/GeneExprPredictor/paper_figures/AFR_EUR_LD_gain_supp.pdf'
    )


if __name__ == '__main__':
    main()