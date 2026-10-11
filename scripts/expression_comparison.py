import pickle
import scipy.stats as stats
import numpy as np

def nonzero_coeff_filter(best_coef, gene, thresh_coeff):
    # Grab nonzero coefficients from best model
    coef_filter = (best_coef > thresh_coeff) | (best_coef < -1*thresh_coeff)
    num_nonzero_coeff = np.sum(coef_filter)

    return num_nonzero_coeff


def evaluate_models(model, true_expr, pred_expr, gene_id):
    # Compute adjusted r2
    r2 = np.corrcoef(true_expr, pred_expr)[0, 1] ** 2
    n = true_expr.shape[0] # number of samples
    coef = model.coef_
    p = nonzero_coeff_filter(
        coef, gene_id, 1e-8
    )
    #adj_r2 = 1 - ((1 - r2)*(n-1)/(n-p-1))

    # R-squared confidence interval
    CI_95 = np.sqrt(4*r2*((1-r2)**2)*((n-p-1)**2)/((n**2-1)*(n+3)))
    #print(f"Gene: {gene_id} rsq: {r2} CI 95%:{CI_95}")
    return r2, CI_95


def main():
    output_file = "/expanse/projects/gymreklab/mlamkin/GeneExprPredictor_data/to_remove.txt"
    #output_file = "/expanse/projects/gymreklab/mlamkin/GeneExprPredictor_data/GTEx_predicted_results.txt"
    list_of_genes = "/expanse/projects/gymreklab/mlamkin/GeneExprPredictor_data/gene_sets/gtex/gtex_pred_genes.txt"
    with open(list_of_genes, 'r') as gene_data:
        with open(output_file, 'w') as output:
            # Write table header
            output.write("gene_name\tgene_id\tall_sample_snp_r2\tall_snp_CI\tall_sample_snp_tr_r2\tall_snp_tr_CI\teur_sample_snp_r2\teur_snp_CI\teur_sample_snp_tr_r2\teur_snp_tr_CI\tafr_sample_snp_r2\tafr_snp_CI\tafr_sample_snp_tr_r2\tafr_snp_tr_CI\n")
            for gene in gene_data:
                split_gene = gene.strip().split('\t')
                gene_name = split_gene[0]
                gene_id = split_gene[1]

                # TODO remove
                if not gene_name == 'KIAA1274': continue

                print(f"{gene_name}")

                gene_sample_info = "/expanse/projects/gymreklab/mlamkin/GeneExprPredictor_data/model_results/predicted_gene_expression/gtex/Cells_EBV-transformed_lymphocytes/Cells_EBV-transformed_lymphocytes.v8.subset_samples.txt"

                # Load gene models to grab nonzero coeffs
                snp_model_file = f"/expanse/projects/gymreklab/mlamkin/GeneExprPredictor_data/model_results/saved_models/geuvadis/{gene_id}_SNPs.pkl" 
                snp_tr_model_file = f"/expanse/projects/gymreklab/mlamkin/GeneExprPredictor_data/model_results/saved_models/geuvadis/{gene_id}_SNPs+STRs.pkl"
                nonlin_model_file = f"/expanse/projects/gymreklab/mlamkin/GeneExprPredictor_data/model_results/saved_models/geuvadis/{gene_id}_SNPs+STRs+NonlinearSTRs.pkl"

                with open(snp_model_file, 'rb') as snp_f:
                    snp_model = pickle.load(snp_f)
                with open(snp_tr_model_file, 'rb') as snp_tr_f:
                    snp_tr_model = pickle.load(snp_tr_f)
                with open(nonlin_model_file, 'rb') as nonlin_f:
                    nonlin_model = pickle.load(nonlin_f)

                # Load header data to get index of gene_id
                header_file = "/expanse/protected/gymreklab-dbgap/mount/mlamkin/GTEx/GTEx_Analysis_v8_eQTL_expression_matrices/Cells_EBV-transformed_lymphocytes.v8.normalized_expression_subset_sample.bed.gz"
                header_data = np.loadtxt(header_file, dtype=np.str_)[:,:4]
                split_genes = np.char.split(header_data[:,3], sep='.') # Creates numpy array of lists that we need to convert
                split_gene_ids = np.array([split_genes[i][0] for i in range(split_genes.shape[0])], dtype=object)

                # Load sample population info
                GTEx_sample_pop_file = "/expanse/protected/gymreklab-dbgap/mount/mlamkin/GTEx/GTEx_Analysis_v8_eQTL_expression_matrices/Cells_EBV-transformed_lymphocytes.v8.pop_info.txt"
                gtex_sample_pop = np.loadtxt(GTEx_sample_pop_file, dtype=object)

                # Load Predicted expression
                try:
                    snp_pred_expr_file = f"/expanse/projects/gymreklab/mlamkin/GeneExprPredictor_data/model_results/predicted_gene_expression/gtex/Cells_EBV-transformed_lymphocytes/{gene_id}_snp_predicted_expr.csv"
                    snp_tr_pred_expr_file = f"/expanse/projects/gymreklab/mlamkin/GeneExprPredictor_data/model_results/predicted_gene_expression/gtex/Cells_EBV-transformed_lymphocytes/{gene_id}_snp_tr_predicted_expr.csv"
                    nonlin_pred_expr_file = f"/expanse/projects/gymreklab/mlamkin/GeneExprPredictor_data/model_results/predicted_gene_expression/gtex/Cells_EBV-transformed_lymphocytes/{gene_id}_nonlinear_predicted_expr.csv"

                    snp_pred_expr = np.loadtxt(snp_pred_expr_file)
                    snp_tr_pred_expr = np.loadtxt(snp_tr_pred_expr_file)
                    nonlin_pred_expr = np.loadtxt(nonlin_pred_expr_file)
                except:
                    print("No Expression Generated")
                    continue

                # Load True Expression (sample x gene)
                true_expr_file = "/expanse/protected/gymreklab-dbgap/mount/mlamkin/GTEx/GTEx_Analysis_v8_eQTL_expression_matrices/Cells_EBV-transformed_lymphocytes.v8.normalized_filtered_final_residuals_subset_sample.csv"
                true_expr = np.loadtxt(true_expr_file)
                gene_true_expr = true_expr[:,split_gene_ids == gene_id].flatten()

                assert gene_true_expr.shape == snp_pred_expr.shape
                assert gene_true_expr.shape == snp_tr_pred_expr.shape

                EUR_mask = gtex_sample_pop[:,1] == 'EUR'
                EUR_gene_true_expr = gene_true_expr[EUR_mask]
                EUR_snp_pred_expr = snp_pred_expr[EUR_mask]
                EUR_snp_tr_pred_expr = snp_tr_pred_expr[EUR_mask]
                EUR_nonlin_pred_expr = nonlin_pred_expr[EUR_mask]

                AFR_mask = gtex_sample_pop[:,1] == 'AFR'
                AFR_gene_true_expr = gene_true_expr[AFR_mask]
                AFR_snp_pred_expr = snp_pred_expr[AFR_mask]
                AFR_snp_tr_pred_expr = snp_tr_pred_expr[AFR_mask]
                AFR_nonlin_pred_expr = nonlin_pred_expr[AFR_mask]

                # Output results in a table rows = genes cols = r-sq comparisons for each pop unadjusted
                #print("All Samples (SNP, SNP+TR, Nonlinear)")
                all_snp_r2, all_snp_CI = evaluate_models(snp_model, gene_true_expr, snp_pred_expr, gene_id)
                all_snp_tr_r2, all_snp_tr_CI = evaluate_models(snp_tr_model, gene_true_expr, snp_tr_pred_expr, gene_id)
                #all_nonlin_r2, all_nonlin_CI = evaluate_models(nonlin_model, gene_true_expr, nonlin_pred_expr, gene_id)

                #print("EUR Samples (SNP, SNP+TR, Nonlinear)")
                eur_snp_r2, eur_snp_CI = evaluate_models(snp_model, EUR_gene_true_expr, EUR_snp_pred_expr, gene_id)
                eur_snp_tr_r2, eur_snp_tr_CI = evaluate_models(snp_tr_model, EUR_gene_true_expr, EUR_snp_tr_pred_expr, gene_id)
                eur_nonlin_r2, eur_nonlin_CI = evaluate_models(nonlin_model, EUR_gene_true_expr, EUR_nonlin_pred_expr, gene_id)

                print(f"EUR snp r2: {eur_snp_r2} EUR SNP+TR r2: {eur_snp_tr_r2} EUR nonlin r2: {eur_nonlin_r2}")

                #print("AFR Samples (SNP, SNP+TR, Nonlinear)")
                afr_snp_r2, afr_snp_CI = evaluate_models(snp_model, AFR_gene_true_expr, AFR_snp_pred_expr, gene_id)
                afr_snp_tr_r2, afr_snp_tr_CI = evaluate_models(snp_tr_model, AFR_gene_true_expr, AFR_snp_tr_pred_expr, gene_id)
                afr_nonlin_r2, afr_nonlin_CI = evaluate_models(nonlin_model, AFR_gene_true_expr, AFR_nonlin_pred_expr, gene_id)

                print(f"AFR snp r2: {afr_snp_r2} AFR SNP+TR r2: {afr_snp_tr_r2} AFR nonlin r2: {afr_nonlin_r2}")

                output.write(f"{gene_name}\t{gene_id}\t{all_snp_r2}\t{all_snp_CI}\t{all_snp_tr_r2}\t{all_snp_tr_CI}\t{eur_snp_r2}\t{eur_snp_CI}\t{eur_snp_tr_r2}\t{eur_snp_tr_CI}\t{afr_snp_r2}\t{afr_snp_CI}\t{afr_snp_tr_r2}\t{afr_snp_tr_CI}\n")
    return

if __name__ == '__main__':
    main()
