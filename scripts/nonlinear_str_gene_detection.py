#!/usr/bin/env python3

"""
Example command:
nohup python3 nonlinear_str_gene_detection.py \
    --chrom chr19 \
    --expr /storage/mlamkin/workspace/TWAS-str/EUR_normalized_filtered_final_residuals_hg38.csv \
    --exprannot /storage/mlamkin/workspace/TWAS-str/EUR_normalized_and_filtered_hg38.csv \
    --covariate /storage/mlamkin/workspace/TWAS-str/igsr-geuvadis.tsv \
    --distfromgene 450 \
    --strgt /gymreklab-tscc/helia/ensembl/ensemble_out/ensemble_chr19_filtered.vcf.gz \
    --gene ENSG00000160213.5 \
    --norm \
    --out /storage/mlamkin/data/TWAS-str/Geuvadis_nonlinear_associations/Geuvadis_chr19_nonlinear_associations.tab &
"""

import sys
import math
import argparse
import numpy as np
import statsmodels.api as sm
from scipy import stats as st
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from haptools.data import GenotypesTR
from cyvcf2 import VCF


EXPRFILE = None
EXPRANNOTFILE = None
DISTFROMGENE = None
STRGTFILE = ""
OUTFILE = ""
CHECKCHROM = False
PERMUTE_EXPR = False
DEBUG = False
MINSAMPLES = 0
MINGENOTYPE = 3

class NonlinearDetector:
    def __init__(
        self,
        EXPRFILE=None,
        EXPRANNOTFILE=None,
        STRGTFILE=None,
        COVFILE=None,
        OUTFILE=None,
    ):
        self.expr_file = EXPRFILE
        self.header_file = EXPRANNOTFILE
        self.str_file = STRGTFILE
        self.metadata_file = COVFILE
        self.outfile = OUTFILE

    def PROGRESS(self, msg):
        sys.stderr.write("%s\n"%msg.strip())

    def GetKeepSamplesAlleles(self, locus_str, gene_expr, mincount):
        """
        locus_str: np.array
            2D genotype array of a single variant samples x strand 
        mincount: int
            Minimum number of each allele that must be present otherwise removed
        """
        # Filter all alleles with unique count < mincount
        uniq, counts = np.unique(locus_str, return_counts=True)
        remove_alleles = uniq[counts < mincount]
        filtered_alleles = np.all(np.isin(locus_str, remove_alleles, invert=True), axis=1)        
        locus_str = locus_str[filtered_alleles,:]
        gene_expr = gene_expr[filtered_alleles]

        # Filter out alleles > 250 since those are empty genotypes encoded
        empty_gts = (locus_str > 250).any(axis=1)
        locus_str = locus_str[~empty_gts,:]
        gene_expr = gene_expr[~empty_gts]

        return locus_str, gene_expr

    def ZNorm(self, vals, m=None, sd=None):
        if m is None: m = np.mean(vals)
        if sd is None: sd = math.sqrt(np.var(vals))
        if sd == 0: return None
        return [(item-m)/sd for item in vals]
    
    def SplitData(self, x, y, train_size):
        """
        Split the data into test and training sets in accordance with the allele distribution.
        Arguments
            x - data of format (n_samples, n_features)
            y - output attempting to be associated with x (n_samples, n_outputs)
            train_size - Fraction of the dataset which will be attributed to train set
        Return
            x_train - x data split into train set
            x_test - x data split into test set
            y_train - y data split into train set
            y_test - y data split into test set
        """
        assert 0 < train_size < 1
        alleles  = x.columns
        x["y"] = y
        swap = (x['A1'] > x['A2'])
        x.loc[swap, ['A1', 'A2']] = x.loc[swap, ['A2', 'A1']].values
        n_samples = x.shape[0]
        
        # get distribution of tuples
        allele_dist = x.groupby(['A1', 'A2']).size().to_frame('freqs')
    
        # list of indices
        train = []
        test = []
    
        # keep track of train allele distribution
        train_freqs = {}
        for ind in allele_dist.index:
            train_freqs[ind] = 0
        
        # assign indices to each set
        for idx, row in x.iterrows():
            a1 = row['A1']
            a2 = row['A2']
            threshold = math.floor(train_size * allele_dist.loc[(a1, a2),'freqs'])
            # Train on more variable data than test with it if there is only a small number
            if threshold < 1: threshold = 1
            if train_freqs[(a1, a2)] < threshold:
                train_freqs[(a1, a2)] += 1
                train.append(idx)
            else:
                test.append(idx)
        return x.loc[train,alleles], x.loc[test,alleles], x.loc[train,'y'], x.loc[test,'y']

    def RandomForestRegression(self, x_train, x_test, y_train, y_test):
        """
        Random Forest Regression model trained via x_train and y_train and evaluated with x_test and y_test.
        Arguments
            x_train - Dataframe for train set where indices are STR loci cols: [gt, A1, A2] = [Overall Genotype, Allele1, Allele2]
            x_test - Dataframe for test setwhere indices are STR loci cols: [gt, A1, A2] = [Overall Genotype, Allele1, Allele2]
            y_train - array of corresponding expression values to genotypes in x_train shape: (train samples, 1)
            y_test - array of corresponding expression values to genotypes in x_test shape: (test samples, 1)
        Return
            r_sq - R squared value of the model created via predictions on x_test and truth of y_test
            rsq_pvalue - pvalue reported from t-test of R squared value to indicate significance of signal
        """
        # Need to format X data so it is (n_samples, n_features)
        # update features so order of alleles is not important (A1+A2, A1-A2)
        for df in [x_train, x_test]:
            df["A1+A2"] = df["A1"] + df["A2"]
            df["|A1-A2|"] = abs(df["A1"] - df["A2"])

        x_train = x_train[["A1+A2", "|A1-A2|"]].to_numpy()
        x_test = x_test[["A1+A2", "|A1-A2|"]].to_numpy()
    
        # Ensure test and train data has variability and isnt constant
        var_a1, var_a2 = np.var(x_test, axis=0)
        if var_a1==0 and var_a2==0: return [None]*2

        var_b1, var_b2 = np.var(x_train, axis=0)
        if var_b1==0 and var_b2==0: return [None]*2

        # These parameters were created via an initial randomized search on a broader scope of parameters
        param_grid = {
            'max_depth': [6, 7, 8, 9],
            'min_samples_leaf': [3, 4, 5],
            'min_samples_split': [3, 5, 7],
            'n_estimators': [30, 40, 50]
        }

        # Base model
        rf = RandomForestRegressor()

        # Run grid search to optimize parameters
        rf_model = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=10)

        # Fit the grid search to the data and predict test regression values
        rf_model.fit(x_train, y_train)
        y_pred = rf_model.predict(x_test)
        n = len(y_test)
        r_value, r_pvalue = st.pearsonr(y_pred, y_test)
        r_sq = r_value**2
        # n is population size and t is the test statistic used to calculate a pvalue
        n = len(y_test)
        # tstat = r*sqrt(n-2)/sqrt(1-r^2)
        t_stat = r_value * np.sqrt(n-2)/np.sqrt(1-r_sq)
        rsq_pvalue = st.t.sf(np.abs(t_stat), n-1)*2 # two-sided pvalue
        return r_sq, rsq_pvalue

    def QuadraticRegression(
        self, 
        X, 
        Y, 
        norm=False, 
        translate_alleles=None, 
        minsamples=0,
    ):
        # update format of X and Y
        # X has format [constant, linear variable, quadratic variable]
        X, Y = self.QuadFormat(X, Y, norm, translate_alleles, minsamples)
        if X is None: return [None]*8

        # create quadratic model
        res_ols = sm.OLS(Y, X, missing="drop").fit()

        # grab parameters
        b1, b2 = res_ols.params[1:]
        b1_se, b2_se = res_ols.bse[1:]
        b1_p, b2_p = res_ols.pvalues[1:]
        f_pval = res_ols.f_pvalue
        return res_ols, b1, b1_se, b1_p, b2, b2_se, b2_p, f_pval

    def QuadFormat(self, X, Y, norm=False, translate_alleles=None, minsamples=0):
        if norm:
            alleles = np.sum(X, axis=1)
            X1 = self.ZNorm(X[:,0], m=np.mean(alleles), sd=math.sqrt(np.var(alleles)))
            X2 = self.ZNorm(X[:,1], m=np.mean(alleles), sd=math.sqrt(np.var(alleles)))
            Y = self.ZNorm(Y)
        else:
            X1 = X[:,0]
            X2 = X[:,1]
        if translate_alleles is not None:
            X1 = X1 + translate_alleles
            X2 = X2 + translate_alleles

        if X1 is None or X2 is None or Y is None: return [None]*2
        if np.var(X1)==0 and np.var(X2)==0: return [None]*2
        if len(X1)<minsamples: return [None]*2
        X_lin = np.sum(X, axis=1)
        X_quad = np.sum(np.square(X), axis=1)
        X = np.column_stack((X_lin, X_quad))
        # add constant
        X = np.column_stack((np.ones(len(X)), X))
        return X, Y

    def LinearRegression(self, X, Y, norm=False, minsamples=0):
        # convert X and Y to proper format to be fitted by OLS
        # X has format [constant, linear variable]
        X, Y = self.LinearFormat(X, Y, norm, minsamples)
        if X is None: return [None]*4

        # Create and fit the linear regression model
        res_ols = sm.OLS(Y, X, missing='drop').fit()

        # Since constant is first value we take second for STR variable
        pval = res_ols.pvalues[1]
        slope = res_ols.params[1]
        err = res_ols.bse[1]
        return res_ols, slope, err, pval

    def LinearFormat(self, X, Y, norm=False, minsamples=0):
        """
        Update the format of X and Y to create the linear regression model. X = [const, x1+x2] Y = [list of gene expr vals]
        Arguments
            X - Dataframe with columns [gt, A1, A2] = [genotype, Allele1, Allele2]
            Y - Gene expression values corresponding to Genotypes/Alleles in X
            norm - Is the data normalized True or False
            minsamples - minimum number of samples required before creating the model
        Return
            X - Properly formatted Dataframe of Alleles to create linear regression model
            Y - Properly formatted list of gene expression values 
        """
        # Only use sum of alleles
        X = np.sum(X, axis=1)
        if norm:
            X = self.ZNorm(X)
            Y = self.ZNorm(Y)
        if X is None or Y is None: return [None]*2
        if np.var(X)==0: return [None]*2
        if len(X) < minsamples: return [None]*2
        # add constant for lin reg model
        X = np.vstack([np.ones(len(X)), X]).T
        return X, Y

    def EvaluateRegression(self, model, x_test, y_test, model_type):
        """
        Given either the quadratic or linear regression model evaluate its performance via R squared
        of predicted gene expression values and true gene expreesion values.
        Arguments
            model - Linear or quadratic model already fit
            x_test - Dataframe of format [gt, A1, A2] = [Genotype, Allele1, Allele2]. This is a subset of the larger dataset
            y_test - List of gene expression values corresponding to each sample in x_test
            model_type - "Linear" or "Quadratic" to indicated how the data needs to be formatted for predicting
        Return
            r_sq - R squared values of predicted gene expression values (y_pred) and true gene expression values (y_test)
            rsq_pvalue - p value from pearson correlation R squared used for identifying signal strength
        """
        if model is None: return [None]*2
        # determine what type of data format the x_test needs to be converted too for prediction
        if model_type == "Linear":
            x_test, y_test = self.LinearFormat(x_test, y_test)
        elif model_type == "Quadratic":
            x_test, y_test = self.QuadFormat(x_test, y_test)
        else:
            return [None]*2
        
        # catch if formatting failed
        if x_test is None or y_test is None: return [None]*2

        # predict gene expression values
        y_pred = model.predict(x_test)

        # get r^2 of predicted gene expression vs true gene expression
        r_value, p_value = stats.pearsonr(y_pred, y_test)
        r_sq = r_value**2
        # n is population size and t is the test statistic used to calculate a pvalue
        n = len(y_test)
        # tstat = r*sqrt(n-2)/sqrt(1-r^2)
        t_stat = r_value * np.sqrt(n-2)/np.sqrt(1-r_sq)
        rsq_pvalue = stats.t.sf(np.abs(t_stat), n-1)*2 # two-sided pvalue
        return r_sq, rsq_pvalue

    # requres the regression results returned from sm.OLS(Y, X, missing='drop').fit() typically res_ols
    def RegressionANOVA(self, model1, model2):
        """
        Perform an ANOVA on two regression models.
        Arguments
            model1 - model to compare against model2 typically the Linear Regression model
            model2 - model to compare against model1 typically the Quadratic Regression model
        Return
            pval - pvalue that the second model is significant from the first model
        """
        table = sm.stats.anova_lm(model1, model2, typ=1)
        pval = table["Pr(>F)"][1]
        if math.isnan(pval):
            pval = None
        return pval

    def filter_maf(self, gts, threshold=0.1):
        # filter maf 
        num_strands = 2 * gts.data.shape[0]
        common_variants = np.zeros((gts.data.shape[1],)).astype(np.bool_)
        # NOTE Uses Heterozygosity measure (1 - sum_i (freqs_i^2)) instead of MAF
        for variant in range(gts.data.shape[1]):
            v_data = gts.data[:, variant, :2]
            alleles, counts = np.unique(v_data, return_counts=True)
            sum_freq_sq = np.sum(np.square(counts/num_strands))
            heterozygosity = 1 - sum_freq_sq
            common_variants[variant] = heterozygosity > threshold
        gts.data = gts.data[:,common_variants,:]
        gts.variants = gts.variants[common_variants]
        return gts

    def prepare_header_data(self, sample_start_ind, gene_ind, chrom=None, gene=None):
        # Collect samples names and gene names from original expression file EUR_normalized_and_filtered_hg38.csv
        header_data = np.loadtxt(self.header_file, delimiter="\t", comments=None, dtype=str)
        all_samples = list(header_data[0][sample_start_ind:])
        all_genes = header_data[1:,gene_ind]

        if gene:
            genes = [gene]
        elif chrom:
            genes = header_data[header_data[:,0] == chrom, gene_ind]
        else:
            genes = header_data[1:,gene_ind]
        del header_data

        return all_samples, all_genes, genes

    def collect_gene_region(self, gene_name, dist, sample_ind, gene_ind):
        """
        Grab region spanning gene with a distance upstream and downstream from 
        gene region of length dist.
        eg. gene coordinates are chr10:1000000-1010000 and dist is 200kb
            region = chr10:800000-1210000

        Parameters
        ----------
        gene_name: str
            Name of gene to collect coordinates from header file
        dist: float
            Distance in kilobases upstream and downstream of gene to collect SNPs/STRs from

        Returns
        -------
        chrom: str
            chromosome of region used for grabbing correct file. eg. chr10
        region: str
            region to collect SNPs and STRs from eg. chr10:800000-1210000
        """
        header_data = np.loadtxt(self.header_file, delimiter="\t", comments=None, dtype=str)[1:,:sample_ind]
        gene_data = header_data[header_data[:,gene_ind] == gene_name].flatten()
        chrom = gene_data[0]
        cis_dist = dist*1000
        region = f"{chrom}:{max(int(gene_data[2]) - cis_dist, 1)}-{int(gene_data[2]) + cis_dist}"
        return chrom, region

    def prepare_gene_data(
        self,
        gene_name,
        genes,
        samples,
        region=None,
        str_variants=None,
        het_thresh=0.1
    ):
        # Load STR Genotypes
        str_gts = GenotypesTR(self.str_file, vcftype='hipstr')
        # Check if the region is empty in the vcf # TODO if haptools updates and returns a null array use that instead
        variants = VCF(self.str_file, samples=samples)(region)
        try:
            var = next(variants)
        except StopIteration:
            print(f"No Variants found in region {region}")
            return None, None 

        del variants

        str_gts.read(region=region, samples=samples, variants=str_variants)
        str_gts = self.filter_maf(str_gts, het_thresh)
        str_gts.subset(samples=samples, inplace=True)

        # NOTE gene expression data normalized, filtered, and PEER factor residuals removed
        # data in the shape of samples x genes
        expr_data = np.loadtxt(self.expr_file, delimiter=" ", dtype=np.float32)
        gene_expr_data = expr_data[:, genes == gene_name].flatten()
        subset_samples = np.where(np.isin(samples, str_gts.samples))[0]
        gene_expr_data = gene_expr_data[subset_samples]

        return gene_expr_data, str_gts
    
    def detect_effects(
        self,
        norm,
        gene_col,
        sample_ind,
        gene=None,
        chrom=None,
        minsamples=0,
        gene_dist=450,
        mingenotype=3
    ):
        # Write header
        f = open(self.outfile, "w")
        header_items = ["gene","chrom","str.id","str.start","n.samples","beta",
                        "beta.se","p.linear","b1","b1.se","p.b1","b2",
                        "b2.se","p.b2","p.quad","p.anova"]
        f.write("\t".join(header_items)+"\n")

        # Load samples, gene names to iterate over, and covariates for calculating residuals
        eur_samples, all_genes, gene_names = self.prepare_header_data(sample_ind, gene_col, chrom, gene)

        # Find associations across genes
        for gene_name in gene_names:
            # Load gene expression and all str_gts in the region
            # NOTE gene expression data normalized, filtered, and PEER factor residuals removed
            cur_chrom, gene_region = self.collect_gene_region(gene_name, gene_dist, sample_ind, gene_col)
            print(f"Generating associations for {gene_name}, all variants within {gene_region}")
            gene_expr_res, all_str_gts = self.prepare_gene_data(
                gene_name,
                all_genes,
                eur_samples,
                region=gene_region,
            )

            if all_str_gts is None:
                continue

            total_samples = all_str_gts.data.shape[0]
            print(f"Performing testing on {total_samples} samples")

            # Iterate over all STRs to perform testing
            for j in range(all_str_gts.data.shape[1]):
                # Gather a single STR and filter all NA within
                locus_str_gts = all_str_gts.data[:, j, :]
                invalid_alleles = np.all(locus_str_gts < 254, axis=1)
                locus_str_gts = locus_str_gts[invalid_alleles,:]
                filtered_gene_expr_res = gene_expr_res[invalid_alleles]

                # Grab STR ID and start position from all_str_gts.variants
                cur_id = all_str_gts.variants["id"][j]
                cur_pos = all_str_gts.variants["pos"][j]

                # Filter outlier alleles (seen < 3 times)
                locus_str_gts, filtered_gene_expr_res = self.GetKeepSamplesAlleles(
                    locus_str_gts, filtered_gene_expr_res, mingenotype
                )
                
                # Run regression using Linear and Quadratic on full data for anova comparison/p-values
                mod_lin, beta, beta_se, p_linear = self.LinearRegression(
                    locus_str_gts,
                    filtered_gene_expr_res,
                    norm=norm,
                    minsamples=minsamples,
                    )

                # our check to determine whether we should consider a STR
                if beta is None: continue

                min_len = np.min(locus_str_gts)
                mod_quad, b1, b1_se, b1_p, b2, b2_se, b2_p, p_quad = self.QuadraticRegression(
                    locus_str_gts,
                    filtered_gene_expr_res,
                    norm=norm,
                    translate_alleles=min_len,
                    minsamples=minsamples,
                )
                # split data into train and test data 
                #x_train, x_test, y_train, y_test = SplitData(locus_str, locus_y["expr"].values, train_size=TRAIN)
                
                # Get an rscore from x_test and y_test which will be compared against random forest
                #lin_rsq, lin_r_pval = EvaluateRegression(mod_lin, x_test, y_test, "Linear")
                #quad_rsq, quad_r_pval = EvaluateRegression(mod_quad, x_test, y_test, "Quadratic")

                # Random forest model and evaluation with rsquared score
                #rf_rsq, rf_r_pval = [None]*2
                #if RF:
                #    rf_rsq, rf_r_pval = RandomForestRegression(x_train, x_test, y_train, y_test)

                # collect r squared scores and convert None to zero for comparison below
                #rsq_scores = []
                #for rscore, model_type in zip([lin_rsq, quad_rsq, rf_rsq], ["Linear", "Quad", "Random_Forest"]):
                #    if rscore is None: 
                #        rscore = 0
                #    rsq_scores.append((rscore, model_type))

                # Run an ANOVA test against the linear and quadratic models
                # (linear model, quadratic model) if significant p-value implies the quadratic model is more suited for the data 
                anova_p = None
                if mod_lin and mod_quad:
                    anova_p = self.RegressionANOVA(mod_lin, mod_quad)

                # Write output
                items = [gene_name, cur_chrom, cur_id, cur_pos, locus_str_gts.shape[0], beta, beta_se, p_linear]
                items.extend([b1, b1_se, b1_p, b2, b2_se, b2_p, p_quad, anova_p])
                f.write("\t".join([str(item) for item in items])+"\n")
        print("STR Association Complete!")
        f.close()

def main():
    parser = argparse.ArgumentParser(description="get datasets for LMM assocation tests with STRs and SNPs")
    parser.add_argument("--expr", help="Normalized expression residuals", type=str, required=True)
    parser.add_argument("--exprannot", help="Header file for gene expression residuals. Contains gene names and locations and sample names.", type=str, required=True)
    parser.add_argument("--covariate", help="Covariate file for calculating residuals", type=str, required=False)
    parser.add_argument("--chrom", help="Restrict analysis to this chromosome", type=str, required=False)
    parser.add_argument("--distfromgene", help="Look at STRs/SNPs within this distance of gene boundaries", type=int, required=True)
    parser.add_argument("--strgt", help="File with STR genotypes. Must be tabix indexed", type=str, required=True)
    parser.add_argument("--out", help="Write results to this file", type=str, required=True)
    parser.add_argument("--gene-col-ind", help="(0 Based) Column number that contains gene names in annotation file", type=int, required=True)
    parser.add_argument("--sample-ind", help="(0 Based) Column number where the sample names begin in the header file.", type=int, required=True)
    parser.add_argument("--permute", help="Permute expression values", action="store_true")
    parser.add_argument("--norm", help="Normalize genotypes before doing association", action="store_true")
    parser.add_argument("--gene", help="Limit analysis to this gene. ex: ENSG00000163576.13", type=str, default=None)
    parser.add_argument("--random-forest", help="Use random forest regression to find association", action="store_true")
    parser.add_argument("--min-samples", help="Require data for this many samples", type=int, default=0)
    parser.add_argument("--min-genotypes", help="Require this many genotypes per test", type=int, default=3)
    parser.add_argument("--trainsize", help="Fraction of samples to place in training set, remaining go to test set. 0 < trainsize < 1", type=float, default=0.7)
    parser.add_argument("--debug", help="Print debug info", action="store_true")

    args = parser.parse_args()
    EXPRFILE = args.expr
    EXPRANNOTFILE = args.exprannot
    COVFILE = args.covariate
    CHROM = args.chrom
    if ("chr" not in str(CHROM)) and CHROM is not None: CHROM="chr%s"%CHROM
    DISTFROMGENE = args.distfromgene
    STRGTFILE = args.strgt
    OUTFILE = args.out
    GENECOL = args.gene_col_ind
    SAMPLEIND = args.sample_ind
    NORM = args.norm
    GENE = args.gene
    RF = args.random_forest
    MINSAMPLES = args.min_samples
    MINGENOTYPE = args.min_genotypes
    TRAIN = args.trainsize


    # Setup Nonlinear Detector
    # Needed files: Expression file, Covariate file, Expression header file for chrom and position and name etc, Genotype file, Output file
    nonlinear_detector = NonlinearDetector(
        EXPRFILE,
        EXPRANNOTFILE,
        STRGTFILE,
        COVFILE,
        OUTFILE,
    )

    # Run nonlinear detection and output results
    nonlinear_detector.detect_effects(
        gene=GENE,
        gene_col=GENECOL,
        sample_ind=SAMPLEIND,
        norm=NORM,
        chrom=CHROM,
        minsamples=MINSAMPLES,
        gene_dist=DISTFROMGENE,
        mingenotype=MINGENOTYPE,
    )
    return

if __name__ == '__main__':
    main()
