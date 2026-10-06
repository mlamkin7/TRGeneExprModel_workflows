import json
import pickle
import scipy.stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import linear_model
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from haptools.data import GenotypesTR, GenotypesVCF
from gene_model_pred import GeneExpressionPredictor

header_file = "/gymreklab-tscc/mlamkin/GeneExprPredictor_data/train_data/EUR_normalized_and_filtered_hg38.csv"
expr_file = "/gymreklab-tscc/mlamkin/GeneExprPredictor_data/train_data/EUR_normalized_filtered_final_residuals_hg38.csv"
snp_file = "/gymreklab-tscc/mlamkin/GeneExprPredictor_data/SNP_vcfs/SNP_{chrom}_filtered.vcf.gz"
str_file = "/gymreklab-tscc/mlamkin/GeneExprPredictor_data/STR_vcfs/ensemble_{chrom}_filtered.vcf.gz"

def compute_adjusted_rsq(n, model, r2):
    # Compute adjusted R-squared = (1 - ((1-r2)*(n-1)/(n-p-1))) n = number of samples, p = number of nonzero coeffs
    coef = model.coef_
    p = len(coef[coef > 0])
    adj_r2 = 1 - ((1 - r2)*(n-1)/(n-p-1))
    return adj_r2

def filter_maf(gts, threshold, str_allele_filter=False):
    # filter maf 
    num_strands = 2 * gts.data.shape[0]
    common_variants = np.zeros((gts.data.shape[1],)).astype(np.bool_)
    # NOTE Uses Heterozygosity measure (1 - sum_i (freqs_i^2)) instead of MAF
    for variant in range(gts.data.shape[1]):
        v_data = gts.data[:, variant, :2]
        alleles, counts = np.unique(v_data, return_counts=True)

        # Remove variants with only 2 alleles
        if str_allele_filter:
            if np.shape(alleles)[0] < 3:
                continue

            # Ensure if over 2 alleles there are at least 3 samples
            if np.sum(counts > 2) < 3:
                continue

        # Calculate heterozygosity measure
        sum_freq_sq = np.sum(np.square(counts/num_strands))
        heterozygosity = 1 - sum_freq_sq
        common_variants[variant] = heterozygosity > threshold

    # Remove variants that don't pass thresholds
    gts.data = gts.data[:,common_variants,:]
    gts.variants = gts.variants[common_variants]
    return gts

def load_genotypes(region, chrom, variant_type='SNP'):
    # Load samples Use same as EUR Geuvadis Samples to make identical to what we train
    header_data = np.loadtxt(
        header_file,
        delimiter="\t",
        dtype=str,
    )
    eur_samples = list(header_data[0][5:])

    if variant_type == 'SNP':
        # Load SNP Genotypes
        snp_single_file = snp_file.format(chrom=chrom)
        snp_gts = GenotypesVCF(snp_single_file)
        snp_gts.read(samples=eur_samples, region=region)
        snp_gts.subset(samples=eur_samples, inplace=True)
        snp_gts = filter_maf(snp_gts, 0.1)
        variants = snp_gts.variants
        lin_snp_gts = snp_gts.data[:,:,:2]
        lin_snp_gts = np.sum(lin_snp_gts, axis=2)

        # Normalize SNP Genotypes
        snp_scaler = StandardScaler()
        norm_lin_snp_gts = snp_scaler.fit_transform(lin_snp_gts)
        return norm_lin_snp_gts, variants
    
    elif variant_type == 'TR':
        # Load TR Genotypes
        str_single_file = str_file.format(chrom=chrom)
        str_gts = GenotypesTR(str_single_file, vcftype='hipstr')
        str_gts.read(region=region, samples=eur_samples)
        str_gts.subset(samples=eur_samples, inplace=True)

        # Filter out repeated TRs
        unique_vars, unique_indices = np.unique(str_gts.variants["pos"], return_index=True)
        str_gts.data = str_gts.data[:, unique_indices, :2]
        str_gts.variants = str_gts.variants[unique_indices]

        # Replace all 255 and 254s 
        converted_str_gts = np.transpose(str_gts.data, (2,0,1))
        str_all_alleles = np.reshape(converted_str_gts,(str_gts.data.shape[0]*str_gts.data.shape[2], str_gts.data.shape[1])) # reshape so pairs of alleles become one col
        allele_modes, counts = scipy.stats.mode(str_all_alleles, axis=0)
        
        # Duplicate modes so theyll work on the 3d array
        allele_modes = np.repeat(allele_modes[:,np.newaxis], repeats=2, axis=1)

        # Filter out non present TR alleles and replace with most common allele
        str_gts.data = np.where(str_gts.data >= 254, allele_modes, str_gts.data)

        # Filter out TRs less than Heterozygosity of 0.1
        str_gts = filter_maf(str_gts, 0.1, str_allele_filter=True)
        lin_tr_gts = str_gts.data[:,:,:2]
        lin_tr_gts = np.sum(lin_tr_gts, axis=2)
        variants = str_gts.variants

        # Normalize TR Genotypes
        tr_scaler = StandardScaler()
        norm_lin_tr_gts = tr_scaler.fit_transform(lin_tr_gts)
        return norm_lin_tr_gts, variants
    
    else:
        raise Exception

def train_models_simulated_data(
    causal_snp_gts,
    causal_tr_gts,
    gts,
    snp_train_gts,
    snp_tr_train_gts,
    num_simulations,
    all_h2,
    h2_start=0,
    step_size=0.1,
    remove_causal_snp=False,
    randomize_samples=False,
):
    # Collect number of causal variants
    num_snps = causal_snp_gts.shape[1]
    num_trs = causal_tr_gts.shape[1]

    sim_rsq_dict = {}
        
    # Update SNP and TR h2
    for tr_h2 in np.arange(h2_start, all_h2+all_h2/10, step_size):
        snp_h2 = all_h2 - tr_h2

        # Have SNP with no impact
        if remove_causal_snp:
            snp_h2 = 0
        
        # Setup dict to store results
        sim_rsq_dict[f'snp_{snp_h2:.2f}_tr_{tr_h2:.2f}'] = []
        sim_rsq_dict[f'adj_snp_{snp_h2:.2f}_tr_{tr_h2:.2f}'] = []

        # Deterministic Effect Sizes
        snp_effects = np.ones((num_snps, num_simulations)) * np.sqrt(snp_h2/num_snps)
        tr_effects = np.ones((num_trs, num_simulations)) * np.sqrt(tr_h2/num_trs)

        # NOTE uncomment if you want randomly distributed effect sizes
        #snp_effects = np.random.normal(loc=0, scale=np.sqrt(snp_h2/num_snps), size=(num_snps, num_simulations))
        #tr_effects = np.random.normal(loc=0, scale=np.sqrt(tr_h2/num_trs), size=(num_trs, num_simulations))
        lin_err = np.random.normal(loc=0, scale=np.sqrt(1-all_h2), size=(gts.shape[0], num_simulations))

        # Generate simulated expression data
        gene_expr_data = causal_snp_gts @ snp_effects + causal_tr_gts @ tr_effects + lin_err

        # Normalize gene expression data
        gt_scaler = StandardScaler()
        gene_expr_data = gt_scaler.fit_transform(gene_expr_data)

        # Train models and evaluate
        paramsearch = GridSearchCV
        max_iter = 50000
        model_params = {"SNPs":{"tol": 1e-4, "max_iter": max_iter},
                "SNPs+STRs":{"tol": 1e-4, "max_iter": max_iter},
                "SNPs+STRs+NonlinearSTRs":{"tol": 1e-4, "max_iter": max_iter}}
        hyperparameters = {"SNPs":{"alpha": np.linspace(0.1, 0.9, 21)},
                "SNPs+STRs":{"alpha": np.linspace(0.1, 0.9, 21)},
                "SNPs+STRs+NonlinearSTRs":{"alpha": np.linspace(0.1, 0.9, 21)}}
        shift_params = {
                        "alpha": {
                            "min_value": 0.001,
                            "shift": 0.09,
                            "num_params": 51,
                        }
                    }
        lasso_model = linear_model.Lasso
        model_types = ["SNPs", "SNPs+STRs"]

        for sim in range(num_simulations):
            print(f"Simulation Number: {sim}")
            sim_predictor = GeneExpressionPredictor(
                paramsearchfxn=paramsearch,
                model=lasso_model,
                model_params=model_params,
                hyperparams=hyperparameters,
                gene_names=[f'snp_{snp_h2:.2f}_tr_{tr_h2:.2f}'],
                sample_ind=None,
                gene_id_ind=None,
                gene_chrom_ind=None,
                gene_start_ind=None,
                gene_end_ind=None,
                train_header_file=None,
                predict_header_file=None,
                snp_file=None,
                str_file=None,
                train_expr_file=None,
                predict_expr_file=None,
                param_shift=shift_params,
                coeff_thresh=None,
                outpath=None,
                model_types=model_types,
                n_models=1,
                n_jobs=1,
                level="ERROR",
            )

            # Train SNP only model
            final_snp_model, snp_r2 = sim_predictor.train_model(
                snp_train_gts,
                gene_expr_data[:,sim],
                gene_name=f'snp_{snp_h2:.2f}_tr_{tr_h2:.2f}_SNP',
                model_params=model_params['SNPs'],
                hyperparameters=hyperparameters['SNPs'],
                title=None,
                cv_figure=None,
            )
            #print(f"Heterozygosity snp: {snp_het} tr: {tr_het} Simulated SNP Only Model Best R-squared: {r2}")

            adj_snp_r2 = compute_adjusted_rsq(snp_train_gts.shape[0], final_snp_model, snp_r2)
            
            # Train SNP+TR Model
            final_snp_tr_model, snp_tr_r2 = sim_predictor.train_model(
                snp_tr_train_gts,
                gene_expr_data[:,sim],
                gene_name=f'snp_{snp_h2:.2f}_tr_{tr_h2:.2f}_SNP_STR',
                model_params=model_params['SNPs+STRs'],
                hyperparameters=hyperparameters['SNPs+STRs'],
                title=None,
                cv_figure=None,
            )

            adj_snp_tr_r2 = compute_adjusted_rsq(snp_tr_train_gts.shape[0], final_snp_tr_model, snp_tr_r2)

            print(f"Unadjusted: {snp_r2} {snp_tr_r2} Adjusted: {adj_snp_r2} {adj_snp_tr_r2}")

            #print(f"Heritability Explained SNP h2: {snp_h2} TR h2: {tr_h2} Simulated SNP+TR Model Best R-squared: {snp_tr_r2}")
            #print(f"Num SNPs {snp_train_gts.shape[1]} Num TRs {snp_tr_train_gts.shape[1] - snp_train_gts.shape[1]}")
            #print("SNP only model")
            #print(np.where(final_snp_model.coef_ > 0.001))
            #print(final_snp_model.coef_[final_snp_model.coef_ > 0.001])
            #print("SNP+TR Model")
            #print(np.where(final_snp_tr_model.coef_ > 0.001))
            #print(final_snp_tr_model.coef_[final_snp_tr_model.coef_ > 0.001])

            # Store simulated R-squareds
            sim_rsq_dict[f'snp_{snp_h2:.2f}_tr_{tr_h2:.2f}'].append((snp_r2, snp_tr_r2))
            sim_rsq_dict[f'adj_snp_{snp_h2:.2f}_tr_{tr_h2:.2f}'].append((adj_snp_r2, adj_snp_tr_r2))

            if randomize_samples:
                # Randomly Select SNPs and TRs
                causal_snp_inds = np.random.randint(snp_train_gts.shape[1], size=num_snps)
                causal_snp_gts = all_snp_gts[:,causal_snp_inds]
                
                tr_gts = snp_tr_train_gts[:,snp_train_gts.shape[1]:]
                causal_tr_inds = np.random.randint(tr_gts.shape[1], size=num_trs)
                causal_tr_gts = tr_gts[:,causal_tr_inds]

                # NOTE uncomment if you want randomly distributed effect sizes
                #snp_effects = np.random.normal(loc=0, scale=np.sqrt(snp_h2/num_snps), size=(num_snps, num_simulations))
                #tr_effects = np.random.normal(loc=0, scale=np.sqrt(tr_h2/num_trs), size=(num_trs, num_simulations))
                lin_err = np.random.normal(loc=0, scale=np.sqrt(1-all_h2), size=(gts.shape[0], num_simulations))

                # Generate simulated expression data
                gene_expr_data = causal_snp_gts @ snp_effects + causal_tr_gts @ tr_effects + lin_err

                # Normalize gene expression data
                gt_scaler = StandardScaler()
                gene_expr_data = gt_scaler.fit_transform(gene_expr_data)
                
    return sim_rsq_dict

def main():
    ### Single Causal Var Simulations ###
    # Load all Genotypes to use as input for the models
    region = 'chr11:57028463-58030803'
    all_snp_gts, all_snp_vars = load_genotypes(region, 'chr11')
    all_tr_gts, all_tr_vars = load_genotypes(region, 'chr11', variant_type='TR')
    all_gts = np.concatenate((all_snp_gts, all_tr_gts), axis=1)

    # Calculate LD between TRs and SNPs and grab the max value
    ld_matrix = np.corrcoef(np.transpose(all_gts))**2
    tr_correlations = ld_matrix[all_snp_gts.shape[1]:,:all_snp_gts.shape[1]]
    tr_max_ld_vals = np.max(tr_correlations, axis=1)

    # Randomly grab SNP region from all loaded genotypes to use as the causal SNP
    causal_snp_ind = np.random.randint(all_snp_gts.shape[1])
    print(f"Causal SNP chosen: {all_snp_vars[causal_snp_ind]}")
    causal_snp_gts = all_snp_gts[:,causal_snp_ind].reshape((all_snp_gts.shape[0], 1))

    # 1. LD Correlation simulations. Plot all different TRs within TIMM10 boundary and varying LD correlations
    ld_simulations = {'gain':[], 'LD_correlation':[]}
    for tr_ind in range(all_tr_gts.shape[1]):
        causal_tr_gts = all_tr_gts[:,tr_ind].reshape(-1,1)
        causal_gts = np.concatenate((causal_snp_gts, causal_tr_gts), axis=1)

        # Parameters for Simulation
        rsq_dict = train_models_simulated_data(
            causal_snp_gts,
            causal_tr_gts,
            causal_gts,
            all_snp_gts,
            all_gts,
            num_simulations=1,
            all_h2=0.8,
            h2_start=0.8,
            step_size=0.2,
        )

        gain = rsq_dict['snp_0.00_tr_0.80'][0][1] - rsq_dict['snp_0.00_tr_0.80'][0][0]
        ld_simulations['gain'].append(gain)
        ld_simulations['LD_correlation'].append(tr_max_ld_vals[tr_ind])

    with open("/storage/mlamkin/data/GeneExprPredictor/paper_figures/TIMM10_trs_gain_ld_simulations.json", 'w') as json_file:
        json.dump(ld_simulations, json_file)
 

    # 2. Single TR with no causal SNP simulation
    # Randomly grab SNP region from all loaded genotypes to use as the causal SNP
    causal_snp_ind = np.random.randint(all_snp_gts.shape[1])
    print(f"Causal SNP chosen: {all_snp_vars[causal_snp_ind]}")
    causal_snp_gts = all_snp_gts[:,causal_snp_ind].reshape((all_snp_gts.shape[0], 1))

    # 1. Collect Existing Genotypes from one TR and SNP
    causal_tr_region = "chr11:57528484-57528485"
    causal_tr_gts, tr_variants = load_genotypes(causal_tr_region, chrom='chr11', variant_type='TR')
    causal_gts = np.concatenate((causal_snp_gts, causal_tr_gts), axis=1)

    # 2. Parameters for Simulation
    large_var_sim_rsq_dict = train_models_simulated_data(
        causal_snp_gts,
        causal_tr_gts,
        causal_gts,
        all_snp_gts,
        all_gts,
        num_simulations=100,
        all_h2=0.8,
        remove_causal_snp=True,
    )

    # Save large_var_sim_rsq_dict to file for use later
    with open("/storage/mlamkin/data/GeneExprPredictor/paper_figures/single_causal_var_sim_rsq.json", 'w') as json_file:
        json.dump(large_var_sim_rsq_dict, json_file)

    def collect_snp_tr_regions(gene_id):
        snp_model_file = f"/gymreklab-tscc/mlamkin/GeneExprPredictor_data/model_results/saved_models/geuvadis/{gene_id}_SNPs.pkl"
        snp_tr_model_file = f"/gymreklab-tscc/mlamkin/GeneExprPredictor_data/model_results/saved_models/geuvadis/{gene_id}_SNPs+STRs.pkl"
        variants_file = f"/gymreklab-tscc/mlamkin/GeneExprPredictor_data/model_results/saved_models/geuvadis/{gene_id}_trained_model_variant_info.txt"

        # load the models t
        models = []
        with open(snp_model_file, 'rb') as snp_f:
            snp_model = pickle.load(snp_f)
        with open(snp_tr_model_file, 'rb') as snp_tr_f:
            snp_tr_model = pickle.load(snp_tr_f)

        snp_variants = []
        tr_variants = []
        with open(variants_file, 'r') as var_f:
            var_f.readline() # Skip header
            for line in var_f:
                var_data = line.strip().split('\t')
                if var_data[0] == 'SNP':
                    snp_variants.append(tuple(var_data[2:]))
                else:
                    tr_variants.append(var_data[2])

        # Define a structured dtype
        dt = np.dtype([('pos', np.uint32), ('ref', object), ('alt', object)])
        snp_variants = np.array(snp_variants, dtype=dt)
        tr_variants = np.array(tr_variants, dtype=np.uint32)

        snp_vars = snp_variants[np.abs(snp_model.coef_) > 0.036]
        tr_vars = tr_variants[np.abs(snp_tr_model.coef_[snp_variants.shape[0]:]) > 0.01]
        return snp_vars, tr_vars
    """
    ### Multi Causal Var Simulations ### 
    print("Multi-SNP and TR Simulations")

    # Load all GTs
    region = "chr17:81918376-82942639"
    all_snp_gts, all_snp_vars = load_genotypes(region, 'chr17')
    all_tr_gts, all_tr_vars = load_genotypes(region, 'chr17', variant_type='TR')
    all_gts = np.concatenate((all_snp_gts, all_tr_gts), axis=1)

    # Calculate LD matrix of SNPs and TRs
    ld_matrix = np.corrcoef(np.transpose(all_gts))**2
    snp_correlations = ld_matrix[:all_snp_gts.shape[1],all_snp_gts.shape[1]:]

    # Calculate LD matrix of SNPs and TRs and collect max LD between SNPs and TRs
    snp_tr_max_ld = np.max(snp_correlations, axis=1)
    max_tr_locs = np.argmax(snp_correlations, axis=1)
    snp_low_ld_locs = np.argwhere(snp_tr_max_ld < 0.1)

    # Grab from list randomly selected SNPs and corresponding low LD TRs
    causal_snp_inds = np.random.choice(snp_low_ld_locs.flatten(), replace=False, size=4)
    causal_tr_inds = max_tr_locs[causal_snp_inds]

    print(f"Causal SNPs chosen: {all_snp_vars[causal_snp_inds]}")
    print(f"Causal TRs chosen: {all_tr_vars[causal_tr_inds]}")
    print(f"LD between causal SNPs and TRs: {snp_tr_max_ld[causal_snp_inds]}")

    causal_snp_gts = all_snp_gts[:,causal_snp_inds]
    causal_tr_gts = all_tr_gts[:,causal_tr_inds]
    causal_gts = np.concatenate((causal_snp_gts, causal_tr_gts), axis=1)

    # Randomly choose SNPs and TRs
    #causal_snp_inds = np.random.randint(all_snp_gts.shape[1], size=2)
    # print(f"Causal SNPs chosen: {all_snp_vars[causal_snp_inds]}")
    #causal_snp_gts = all_snp_gts[:,causal_snp_inds]

    # From the causal SNPs grab the TR inds that have the highest LD
    #causal_tr_inds = np.random.randint(all_tr_gts.shape[1], size=2)
    #print(f"Causal TRs chosen: {all_tr_vars[causal_tr_inds]}")
    #causal_tr_gts = all_tr_gts[:,causal_tr_inds]

    # Collect TR GTs from chosen TRs 
    # gene_id = 'ENSG00000169660' # HEXDC
    # snp_variants, tr_variants = collect_snp_tr_regions(gene_id)

    # tr_regions = []
    # for tr_var in tr_variants:
    #     tr_regions.append(f"chr17:{tr_var}-{tr_var+1}")

    # print(f"TR Variants Start Positions: {tr_variants}")

    # # Load TRs
    # causal_tr_gts = None
    # for tr_region in tr_regions:
    #     norm_lin_tr_gts, tr_vars = load_genotypes(tr_region, 'chr17', variant_type='TR')
    #     if causal_tr_gts is None:
    #         causal_tr_gts = norm_lin_tr_gts
    #     else:
    #         causal_tr_gts = np.concatenate((causal_tr_gts, norm_lin_tr_gts), axis=1)
    #causal_gts = np.concatenate((causal_snp_gts, causal_tr_gts), axis=1)

    # 2. Parameters for Simulation
    multi_var_sim_rsq_dict = train_models_simulated_data(
        causal_snp_gts,
        causal_tr_gts,
        causal_gts,
        all_snp_gts,
        all_gts,
        num_simulations=100,
        all_h2=0.8,
        #randomize_samples=True,
    )

    # Save large_var_sim_rsq_dict to file for use later
    with open("/storage/mlamkin/data/GeneExprPredictor/paper_figures/multi_causal_var_low_LD_rsq.json", 'w') as json_file:
        json.dump(multi_var_sim_rsq_dict, json_file)
    """
        
if __name__ == '__main__':
    main()