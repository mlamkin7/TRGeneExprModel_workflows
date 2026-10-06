import os
import json
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon
from haptools.data import Genotypes
from gene_model_pred import GeneExpressionPredictor
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm

# Colors for each plot
snp_model_color = (244/256, 179/256, 1/256)
snp_tr_model_color = (219/256, 16/256, 72/256)
snp_color = (46/256, 37/256, 133/256)
tr_color = (51/256, 117/256, 56/256)
eur_color = tr_color
afr_color = (88/256, 9/256, 79/256)

# NOTE Important so when loading figures into illustrator the text is seen as text
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42 

h2_threshold = 0.05/12605 # 12605 is the number of nonzero genes in our Geuvadis expression table
master_table = "/storage/mlamkin/data/GeneExprPredictor/Geuvadis_master_table.txt"
gtex_table = "/storage/mlamkin/data/GeneExprPredictor/GTEx_predicted_results.txt"
#master_table = "/expanse/projects/gymreklab/mlamkin/GeneExprPredictor_data/final_results/Geuvadis_master_table.txt"
#gusev_EBV_data = "/storage/mlamkin/data/GeneExprPredictor/gusev_results/Gusev_all_genes_EBV-transformed_lymphocytes.txt"


def afr_gt_box_plot(
    eur_gts,
    afr_gts,
    expr,
    afr_expr,
    best_tr_ind, 
    vars, 
    gene_name
):
    # Filter out gts > 500
    tr_gts = eur_gts[:,best_tr_ind]
    mask = tr_gts < 500
    filtered_gts = tr_gts[mask]
    filtered_expr = expr[mask]
    afr_tr_gts = afr_gts[:,best_tr_ind]
    afr_mask = afr_tr_gts < 500
    afr_filtered_gts = afr_tr_gts[afr_mask]
    afr_filtered_expr = afr_expr[afr_mask]

    fig = plt.figure(figsize=(16,12))
    eTR_ax = fig.add_subplot(1,1,1)
    
    # Plot gts vs expression
    unique_eur = np.unique(filtered_gts)
    for gt in unique_eur:
        gt_mask = filtered_gts == gt
        box = eTR_ax.boxplot(
            filtered_expr[gt_mask], 
            positions=[gt-.1], 
            patch_artist=True,
            label='EUR',
            flierprops=dict(markeredgewidth=0),
        )
        colors = [tr_color]
        # Change colors in boxplot
        for entry in ['boxes', 'fliers']:
            for item, color in zip(box[entry], colors):
                if entry == 'fliers':
                    item.set_markerfacecolor(color)
                else:
                    plt.setp(item, color=color)

        for entry in ['whiskers', 'caps']:
            for items, color in zip(zip(box[entry][::2],box[entry][1::2]),colors):
                plt.setp(items, color=color)

    unique_afr = np.unique(afr_filtered_gts)
    for afr_gt in unique_afr:
        gt_mask = afr_filtered_gts == afr_gt
        box = eTR_ax.boxplot(
            afr_filtered_expr[gt_mask], 
            positions=[afr_gt+.1], 
            patch_artist=True,
            label='AFR',
            flierprops=dict(markeredgewidth=0),
        )
        colors = [afr_color]
        # Change colors in boxplot
        for entry in ['boxes', 'fliers']:
            for item, color in zip(box[entry], colors):
                if entry == 'fliers':
                    item.set_markerfacecolor(color)
                else:
                    plt.setp(item, color=color)

        for entry in ['whiskers', 'caps']:
            for items, color in zip(zip(box[entry][::2],box[entry][1::2]),colors):
                plt.setp(items, color=color)

    x_ticks = np.unique(np.concatenate((unique_eur, unique_afr)))
    eTR_ax.set_xticks(x_ticks)
    eTR_ax.set_xticklabels(x_ticks.astype(np.uint16))

    handles, labels = eTR_ax.get_legend_handles_labels()
    eTR_ax.legend([handles[0], handles[-1]], [labels[0], labels[-1]], loc='upper right')

    # Plot AFR Boxes
    eTR_ax.set_title(f'{gene_name} {vars[best_tr_ind][1]} {vars[best_tr_ind][2]} AFR EUR Gene Expr Association')
    eTR_ax.set_xlabel('TR Genotype (TR_1 + TR_2)')
    eTR_ax.set_ylabel('Gene Expression')
    outpath = f'/storage/mlamkin/data/GeneExprPredictor/paper_figures/{gene_name}/'
    fig.savefig(outpath + f'{gene_name}_{vars[best_tr_ind][1]}_{vars[best_tr_ind][2]}_afr_eur_tr_assoc.pdf')
    return


def gt_box_plot(
    gts, 
    expr, 
    best_tr_ind, 
    best_tr_coeff, 
    vars, 
    gene_name
):
    fig = plt.figure(figsize=(16,12))
    eTR_ax = fig.add_subplot(1,1,1)
    # Plot gts vs expression
    # Filter out genotypes > 500
    tr_gts = gts[:,best_tr_ind]
    mask = tr_gts < 500
    filtered_gts = tr_gts[mask]
    filtered_expr = expr[mask]
    for gt in np.unique(filtered_gts):
        gt_mask = filtered_gts == gt
        box = eTR_ax.boxplot(
            filtered_expr[gt_mask], 
            positions=[gt], 
            patch_artist=True,
            flierprops=dict(markeredgewidth=0),
        )
        colors = [tr_color]
        # Change colors in boxplot
        for entry in ['boxes', 'fliers']:
            for item, color in zip(box[entry], colors):
                if entry == 'fliers':
                    item.set_markerfacecolor(color)
                else:
                    plt.setp(item, color=color)

        for entry in ['whiskers', 'caps']:
            for items, color in zip(zip(box[entry][::2],box[entry][1::2]),colors):
                plt.setp(items, color=color)

    eTR_ax.set_title(f'{gene_name} {vars[best_tr_ind][1]} {vars[best_tr_ind][2]} coeff size: {best_tr_coeff:.4f} Gene Expr Association')
    eTR_ax.set_xlabel('TR Genotype (TR_1 + TR_2)')
    eTR_ax.set_ylabel('Gene Expression')
    outpath = f'/storage/mlamkin/data/GeneExprPredictor/paper_figures/{gene_name}/'
    fig.savefig(outpath + f'{gene_name}_{vars[best_tr_ind][1]}_{vars[best_tr_ind][2]}_best_tr_assoc.pdf')
    return

def plot_model_coeffs(
    model_file, 
    gene_expr_predictor, 
    gene_id,
    ax, 
    orientation=1,
    gts=None,
    expr=None,
    gene_name=None,
    afr_gts=None,
    afr_expr=None,
    tr_coeff_file=None
):
    # Load prior trained models
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    coeff = model.coef_

    # Grab all nonzero coefficients
    coef_filter, var_coef_filter, num_nonzero_coeff = gene_expr_predictor.nonzero_coeff_filter(
        coeff, gene_id, 0.01
    )

    # Filtered variants
    num_snps = len(gene_expr_predictor.snp_variants[gene_id])
    num_trs = len(gene_expr_predictor.str_variants[gene_id])
    wanted_vars = num_snps + num_trs
    var_coef_filter = var_coef_filter[:wanted_vars]
    vars = gene_expr_predictor.variants[gene_id][:wanted_vars]
    snp_nonzero_vars = vars[:num_snps][var_coef_filter[:num_snps]]
    tr_nonzero_vars = vars[num_snps:][var_coef_filter[num_snps:]]
    training_boundary = [vars[0][2], vars[-1][2]]
    #nonzero_vars = vars[var_coef_filter] # print this out to know what the nonzero vars are

    # Filtered Coefficients
    snp_coeffs = coeff[:num_snps]
    snp_coeffs = snp_coeffs[coef_filter[:num_snps]]
    tr_coeffs = coeff[num_snps:]
    tr_coeffs = tr_coeffs[coef_filter[num_snps:]]

    # Plot strongest TR coefficient
    if (gts is not None) and (expr is not None):
        # Find TR using tr_coeffs index and plot
        best_tr_inds = np.arange(coeff[num_snps:].shape[0])
        best_tr_inds = best_tr_inds[np.abs(coeff[num_snps:]) > 0] + num_snps

        for best_tr_ind, best_tr_coeff in zip(best_tr_inds, tr_coeffs):
            if tr_coeff_file is not None:
                tr_coeff_file.write(f'{gene_name}\t{gene_id}\t{vars[best_tr_ind][1]}\t{vars[best_tr_ind][2]}\t{best_tr_coeff}\n')

            gt_box_plot(gts, expr, best_tr_ind, best_tr_coeff, vars, gene_name)

            # afr and eur expression plot
            if (afr_gts is not None) and (afr_expr is not None):
                
                afr_gt_box_plot(    
                    gts,
                    afr_gts,
                    expr,
                    afr_expr,
                    best_tr_ind, 
                    vars, 
                    gene_name
                )

    # So if np.sum(var_coef_filter[len(gene_expr_predictor.snp_variants[gene_name]):]) is > 0 then it contains STRs and you can add the additional search
    num_sig_strs = np.sum(var_coef_filter[len(gene_expr_predictor.snp_variants[gene_id]):])
    if num_sig_strs > 0:
        # Collect SNP and TR coords
        snp_coords = [coord[2] for coord in snp_nonzero_vars]
        tr_coords = [coord[2] for coord in tr_nonzero_vars]
        
        # End figure with SNP+STR on -Y-axis
        ax.scatter(snp_coords, orientation*np.abs(snp_coeffs), color=snp_color, s=100)
        ax.scatter(tr_coords, orientation*np.abs(tr_coeffs), color=tr_color, label=f"TR", s=100)
        ax.vlines(snp_coords, 0, orientation*np.abs(snp_coeffs), color='black', linestyle="dashed")
        ax.vlines(tr_coords, 0, orientation*np.abs(tr_coeffs), color='black', linestyle="dashed")
        ax.plot(
            [
                training_boundary[0],
                training_boundary[1]
            ], 
            [0,0], 
            color='black',
        )
        
    else:
        model_coords = [coord[2] for coord in snp_nonzero_vars]
        ax.scatter(model_coords, orientation*np.abs(snp_coeffs), color=snp_color, label=f"SNP", s=100)
        ax.vlines(model_coords, 0, orientation*np.abs(snp_coeffs), color='black', linestyle="dashed")
        ax.plot(
            [
                training_boundary[0],
                training_boundary[1]
            ], 
            [0,0], 
            color='black',
        )
    
    # Set Plot Limits
    ax.set_xlim(training_boundary)

    chrom = vars[0][1]
    return chrom, snp_coeffs, tr_coeffs

def plot_coeff_comparison(
    gene_id, 
    gene_name,
    tss_start,
    tss_end,
    tss_strand,
    top_gene_expr_predictor,
    bot_gene_expr_predictor,
    top_file, 
    bot_file,
    output,
    afr_expr_predictor=None,
    tr_coeff_file=None,
):
    eur_samples, header_genes = top_gene_expr_predictor.prepare_header_info(top_gene_expr_predictor.train_header_file)

    # Collect the variants
    train_dataset, train_expr = top_gene_expr_predictor.collect_gts_gene_expr(
        gene_id,
        500,
        header_genes,
        eur_samples,
        snp_variants=None,
        str_variants=None,
        expr_file=top_gene_expr_predictor.train_expr_file,
        header_file=top_gene_expr_predictor.train_header_file,
    )

    eur_variants = {}
    eur_vars = top_gene_expr_predictor.variants[gene_id][:train_dataset[1].shape[1]]
    for var_ind, var in enumerate(eur_vars):
        id = f"{var[0]}_{var[1]}_{var[2]}"
        eur_variants[id] = var_ind

    if afr_expr_predictor is not None:
        afr_samples, afr_header_genes = afr_expr_predictor.prepare_header_info(afr_expr_predictor.train_header_file)
        afr_dataset, afr_expr = afr_expr_predictor.collect_gts_gene_expr(
            gene_id,
            500,
            afr_header_genes,
            afr_samples,
            snp_variants=top_gene_expr_predictor.snp_variants[gene_id],
            str_variants=top_gene_expr_predictor.str_variants[gene_id],
            expr_file=afr_expr_predictor.train_expr_file,
            header_file=afr_expr_predictor.train_header_file,
            filter_freq=False,
        )

        afr_gts = np.zeros((afr_dataset[1].shape[0], train_dataset[1].shape[1]))
        afr_variants = afr_expr_predictor.variants[gene_id][:afr_dataset[1].shape[1]]
        for afr_ind, afr_var in enumerate(afr_variants):
            id = f"{afr_var[0]}_{afr_var[1]}_{afr_var[2]}"
            afr_gts[:,eur_variants[id]] = afr_dataset[1][:,afr_ind]

    fig = plt.figure(figsize=(16,12))
    ax = fig.add_subplot(1,1,1)
    ax.set_title(f"{gene_name} Coefficient Comparison")
    ax.set_ylabel("Lasso Model Coefficient Value")
        
    # Plot TSS start and end keeping account of strand orientation
    # Find where true gene start is based on strand
    if tss_strand == '+':
        gene_start = tss_start
        gene_end = tss_end
        dir = 1
    else:
        gene_start = tss_end
        gene_end = tss_start
        dir = -1
    
    # Plot an arrow for gene start
    ax.plot([gene_start, gene_start], [-0.02, 0.02], color='black')
    ax.annotate("", xy=(gene_start + (dir*20000), 0.02), xytext=(gene_start, 0.02),
            arrowprops=dict(arrowstyle="->"))

    # Plot end of gene
    ax.plot([gene_end, gene_end], [-0.02, 0.02], color='black')

    # grab coeffs from snp only model
    chrom, snp_coeffs, tr_coeffs = plot_model_coeffs(
        top_file, 
        top_gene_expr_predictor, 
        gene_id,
        ax,
    )
    
    top_coeffs = snp_coeffs
    ax.set_xlabel(f"{chrom} Coords")

    if top_gene_expr_predictor != bot_gene_expr_predictor:
        eur_samples, header_genes = bot_gene_expr_predictor.prepare_header_info(bot_gene_expr_predictor.train_header_file)

        # Collect the variants
        train_dataset, train_expr = bot_gene_expr_predictor.collect_gts_gene_expr(
            gene_id,
            500,
            header_genes,
            eur_samples,
            snp_variants=None,
            str_variants=None,
            expr_file=bot_gene_expr_predictor.train_expr_file,
            header_file=bot_gene_expr_predictor.train_header_file,
        )
    else:
        bot_gene_expr_predictor = top_gene_expr_predictor

    # Get coeff from snp_str model
    chrom, snp_coeffs, tr_coeffs = plot_model_coeffs(
        bot_file, 
        bot_gene_expr_predictor, 
        gene_id,
        ax,
        orientation=-1, # If it will be on top or bottom of figure (-1 = bottom)
        gts=train_dataset[1], # 1 = snp + tr gts (0 = snp only, 2 = snp + tr + tr^2)
        expr=train_expr,
        gene_name=gene_name,
        afr_gts=afr_gts,
        afr_expr=afr_expr,
        tr_coeff_file=tr_coeff_file,
    )
    bot_coeffs = np.concatenate((snp_coeffs, tr_coeffs))
    if bot_coeffs.shape[0] == 0:
        ylim_min = 0.99
    else:
        ylim_min = np.max(np.abs(bot_coeffs))
    if top_coeffs.shape[0] == 0:
        ylim_max=0.99
    else:
        ylim_max = np.max(np.abs(top_coeffs))
    ylim_bound = max(ylim_min, ylim_max)
    ax.set_ylim(-1*ylim_bound-0.01, ylim_bound+0.01)

    ax.legend()
    fig.savefig(output)
    return

def ball_and_stick(gene_id, gene_name, skip_genes=False, tr_coeff_file=None):
    if not os.path.exists(f"/storage/mlamkin/data/GeneExprPredictor/paper_figures/{gene_name}/"):
        os.mkdir(f"/storage/mlamkin/data/GeneExprPredictor/paper_figures/{gene_name}/")
    elif skip_genes:
        return

    # Load gene start and end coordinates
    TSS_file = "/storage/mlamkin/data/GeneExprPredictor/train_data/GRCh38_V38_TSS.tsv"
    TSS_df = pd.read_csv(TSS_file, delimiter='\t', usecols=['strand', 'name2'])
    TSS_df.columns = ['strand', 'gene']
    TSS_gene = TSS_df[TSS_df['gene'] == gene_name]
    strand = '+'
    try:
        strand = TSS_gene['strand'].iloc[0]
    except:
        print(f"Could not find strand orientation for gene: {gene_name} (Assuming '+' Direction)")

    # Load required genotypes and gene expression to collect coefficients
    #path = "/expanse/projects/gymreklab/mlamkin/" # Use for Expanse
    path = "/gymreklab-tscc/mlamkin/" # Use for Snorlax
    EUR_header_file = path + "GeneExprPredictor_data/train_data/EUR_normalized_and_filtered_hg38.csv"
    EUR_expr_file = path + "GeneExprPredictor_data/train_data/EUR_normalized_filtered_final_residuals_hg38.csv"
    AFR_header_file = path + "GeneExprPredictor_data/train_data/AFR_normalized_and_filtered_hg38.csv"
    AFR_expr_file = path + "GeneExprPredictor_data/train_data/AFR_normalized_filtered_final_residuals_hg38.csv"
    snp_file = path + "GeneExprPredictor_data/SNP_vcfs/SNP_{}_filtered.vcf.gz"
    str_file = path + "GeneExprPredictor_data/STR_vcfs/ensemble_{}_filtered.vcf.gz"

    # Load gene start and end
    gene_info_df = pd.read_csv(EUR_header_file, delimiter='\t', usecols=['start', 'end', 'gene_id'])
    gene_info_df['gene_id'] = gene_info_df['gene_id'].str.split('.').str[0]
    gene_info = gene_info_df[gene_info_df['gene_id'] == gene_id]
    gene_start = gene_info['start'].iloc[0] 
    gene_end = gene_info['end'].iloc[0] 

    # Load models
    snp_model_file = f"/gymreklab-tscc/mlamkin/GeneExprPredictor_data/model_results/saved_models/geuvadis/{gene_id}_SNPs.pkl"
    snp_str_model_file = f"/gymreklab-tscc/mlamkin/GeneExprPredictor_data/model_results/saved_models/geuvadis/{gene_id}_SNPs+STRs.pkl"

    sample_ind = 5 # NOTE for GTEx sample_ind is 4
    gene_id_ind = 3
    gene_chrom_ind = 0
    gene_start_ind = 1
    gene_end_ind = 2

    geuv_gene_expr_predictor = GeneExpressionPredictor(
        paramsearchfxn=None,
        model=None,
        model_params=None,
        hyperparams=None,
        gene_names=[gene_name],
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
        gene_names=[gene_name],
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

    plot_coeff_comparison(
        gene_id,
        gene_name,
        gene_start,
        gene_end,
        strand,
        geuv_gene_expr_predictor,
        geuv_gene_expr_predictor,
        top_file=snp_model_file,
        bot_file=snp_str_model_file,
        output=f"/storage/mlamkin/data/GeneExprPredictor/paper_figures/{gene_name}/{gene_name}_model_coefficient_comparison.pdf",
        afr_expr_predictor=afr_gene_expr_predictor,
        tr_coeff_file=tr_coeff_file,
    )

def main():
    # Load Master Table (gene, p.linear, EUR_alleles, EUR_het, snp_adj_r_squared, snp_str_adj_r_squared)
    rsq_df = pd.read_csv(
        master_table, 
        delimiter='\t', 
        usecols=[
            'gene_name',
            'gene',
            'EUR_SNP_adj_R_sq_Mean', 
            'EUR_SNP_STR_adj_R_sq_Mean',
        ]
    )

    #### Genes that trained ####
    # NOTE trained_rsq_df is the finalized df we work with for downstream analyses all others are for comparison purposes
    # Filter out genes that didnt train (negative for both) and report these updated numbers under the filtering
    trained_rsq_df = rsq_df[(rsq_df['EUR_SNP_adj_R_sq_Mean'] > 0) | (rsq_df['EUR_SNP_STR_adj_R_sq_Mean'] > 0)]
    trained_rsq_df.loc[trained_rsq_df['EUR_SNP_adj_R_sq_Mean'] < 0, 'EUR_SNP_adj_R_sq_Mean'] = 0
    trained_rsq_df.loc[trained_rsq_df['EUR_SNP_STR_adj_R_sq_Mean'] < 0, 'EUR_SNP_STR_adj_R_sq_Mean'] = 0
    trained_rsq_df.fillna(0, inplace=True)

    # Drop gene duplicates now that we've matched variants in estr_h2_rsq_df
    trained_rsq_df.drop_duplicates('gene', inplace=True)
    trained_rsq_df.dropna(inplace=True)

    # Create expr and ball and stick plots for all trained genes
    #tr_coeff_output = "/storage/mlamkin/data/GeneExprPredictor/paper_figures/Trained_models_TR_coeffs.tsv"
    tr_coeff_output = "/storage/mlamkin/data/GeneExprPredictor/to_remove.txt"
    with open(tr_coeff_output, 'w') as tr_coeff_file:
        tr_coeff_file.write(f'Gene_Name\tGene_ID\tChrom\tStart\tTR_Model_Coeff\n')
        for row_ind, row_info in trained_rsq_df[['gene_name', 'gene']].iterrows():
            if not row_info['gene_name'] == 'TIMM10': continue
            print(f"{row_info['gene_name']} {row_info['gene']}")
            ball_and_stick(row_info['gene'], row_info['gene_name'], skip_genes=False, tr_coeff_file=tr_coeff_file)


if __name__ == '__main__':
    main()