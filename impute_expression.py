import pickle
import argparse
import numpy as np
from gene_model_pred import GeneExpressionPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gene_id', help="ID of gene to predict gene expression for. Required to grab collect gene region.")
    parser.add_argument('--snp_model_file', help="SNP only model in the form of a pickle file used to impute expression.")
    parser.add_argument('--snp_tr_model_file', help="SNP+TR model in the form of a pickle file used to impute expression.")
    parser.add_argument('--nonlin_model_file', help="SNP+TR+NonlinTR model in the form of a pickle file used to impute expression.")
    parser.add_argument('--header_file', help="Path to header file. Header file is used to collect gene region in order to know where in the VCF to collect variants for model.")
    parser.add_argument('--variants_file', help="File that contains chrom start ref alt for the variants in each model.")
    parser.add_argument('--snp_file', help="VCF file for SNPs for prediction.")
    parser.add_argument('--tr_file', help="VCF file to collect TRs for prediction.")
    parser.add_argument('--sample_file', help="File that has a sample on each row. Can be collected in VCFs via \'bcftools query -l my_vcf.vcf.gz\'")
    parser.add_argument('--snp_scaler_file', help="pkl file that contains scaler for snp model that was scaled during training to be inversely applied on newly made expression data.")
    parser.add_argument('--snp_tr_scaler_file', help="pkl file that contains scaler for snp+tr model that was scaled during training to be inversely applied on newly made expression data.")
    parser.add_argument('--nonlin_scaler_file', help="pkl file that contains scaler for snp+tr+nonlin_tr model that was scaled during training to be inversely applied on newly made expression data.")
    parser.add_argument('--expr_scaler_file', help="pkl file that contains scaler for gene expression was scaled during training to be inversely applied on newly made expression data.")
    parser.add_argument('--outpath', help="Directory to save outputted gene expression.")
    args = parser.parse_args()
    
    sample_ind = 4 # NOTE for GTEx sample_ind is 4 Geuvadis is 5
    gene_id_ind = 3
    gene_chrom_ind = 0
    gene_start_ind = 1
    gene_end_ind = 2

    samples = []
    with open(args.sample_file,'r') as sample_f:
        for line in sample_f:
            samples.append(line.strip())
    samples = tuple(samples)

    geuv_gene_expr_predictor = GeneExpressionPredictor(
        paramsearchfxn=None,
        model=None,
        model_params=None,
        hyperparams=None,
        gene_names=[args.gene_id],
        sample_ind=sample_ind,
        gene_id_ind=gene_id_ind,
        gene_chrom_ind=gene_chrom_ind,
        gene_start_ind=gene_start_ind,
        gene_end_ind=gene_end_ind,
        train_header_file=args.header_file,
        predict_header_file=args.header_file,
        snp_file=args.snp_file,
        str_file=args.tr_file,
        train_expr_file=None,
        predict_expr_file=None,
        param_shift=None,
        coeff_thresh=None,
        outpath=None,
        n_jobs=1,
        level="DEBUG",
    )

    # load the models to predict
    models = []
    with open(args.snp_model_file, 'rb') as snp_f:
        models.append(pickle.load(snp_f))
    with open(args.snp_tr_model_file, 'rb') as snp_tr_f:
        models.append(pickle.load(snp_tr_f))
    with open(args.nonlin_model_file, 'rb') as nonlin_f:
        models.append(pickle.load(nonlin_f))

    # Load scaler files for genotypes and expression
    with open(args.snp_scaler_file, 'rb') as gt_snp_f:
        gt_snp_scaler = pickle.load(gt_snp_f)
    with open(args.snp_tr_scaler_file, 'rb') as gt_snp_tr_f:
        gt_snp_tr_scaler = pickle.load(gt_snp_tr_f)
    with open(args.nonlin_scaler_file, 'rb') as gt_nonlin_f:
        gt_nonlin_scaler = pickle.load(gt_nonlin_f)
    with open(args.expr_scaler_file, 'rb') as expr_f:
        expr_scaler = pickle.load(expr_f)

    snp_variants = []
    tr_variants = []
    with open(args.variants_file, 'r') as var_f:
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

    geuv_gene_expr_predictor.predict_expression(
        models,
        [gt_snp_scaler, gt_snp_tr_scaler, gt_nonlin_scaler],
        expr_scaler,
        snp_variants,
        tr_variants,
        samples,
        args.outpath,
    )


if __name__ == '__main__':
    main()
