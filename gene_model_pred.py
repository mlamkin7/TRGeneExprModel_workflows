from __future__ import annotations
import os
import pickle
import argparse
import numpy as np
import scipy.stats
from collections.abc import Iterable
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import \
    GridSearchCV, RepeatedKFold, cross_validate, RandomizedSearchCV
from haptools.data import Genotypes, GenotypesVCF, GenotypesTR
from haptools.clump import GetOverlappingSamples
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

class GeneExpressionPredictor:
    def __init__(
        self, 
        paramsearchfxn: function,
        model: object,
        model_params: dict,
        hyperparams: dict,
        gene_names: list,
        sample_ind: int,
        gene_id_ind: int,
        gene_chrom_ind: int,
        gene_start_ind: int,
        gene_end_ind: int,
        train_header_file: str,
        snp_file: str,
        str_file: str,
        train_expr_file: str,
        outpath: str,
        predict_header_file: str = None,
        predict_expr_file: str = None,
        param_shift: dict = None,
        scoring: str = 'r2',
        coeff_thresh: float = 1e-8,
        model_types: list = ["SNPs", "SNPs+STRs", "SNPs+STRs+NonlinearSTRs"],
        n_models: int = 1,
        n_jobs: int = 1,
        level: str = "WARNING",
    ):
        """
        Initialize GeneExpressionPredictor instance.

        Parameters
        ----------
        paramsearchfxn: function
            Search function used to find optimal parameters ex: GridSearchCV or RandomizedSearchCV
        model: object
            sklearn model object ex: linear_model.Lasso
        model_params: dict
            Intrinsic params for each model to run
            ex:{"SNPs":{"tol": 1e-5, "max_iter": max_iter},
              "SNPs+STRs":{"tol": 1e-5, "max_iter": max_iter},
              "SNPs+STRs+NonlinearSTRs":{"tol": 1e-5, "max_iter": max_iter}}
        hyperparams: dict
            hyperparameters to tune in the model object using the paramsearchfxn
            {"STR": {"param_name": value}}
        gene_names: list(str)
            list of genes to generate and train models for.
            Must use ENSG ID due to potential repeats such as ZNF625.
            ex : [ENSG00000197540.1]
        sample_ind: int
            index for starting location of samples in header file
        gene_id_ind: int
            index for gene id in header file
        gene_chrom_ind: int
            index for chromosome in header file
        gene_start_ind: int
            index for gene start position in header file
        gene_end_ind: int
            index for gene end position in header file
        train_header_file: str,
            File to collect sample and gene names for training
        snp_file: str
            Format string of a path ex: "/path/to/file/my_file_{}.txt" where {} is 
            filled in with a chromosome. Overall result: "/path/to/file/my_file_chr21.txt"
        str_file: str 
            Format string of a path ex: "/path/to/file/my_file_{}.txt" where {} is 
            filled in with a chromosome. Overall result: "/path/to/file/my_str_file_chr21.txt"
        train_expr_file: str
            File to grab expression for each sample and gene for training models
        outpath: str
            File path used to store the output files generated from training and predicting process
        predict_header_file: str,
            File to collect sample and gene names for predicting
        predict_expr_file: str,
            File to grab expression for each sample and gene for predicting models
        param_shift: dict
            Stores each parameter for each model and the shift for the second pass
            of hyperparameter search
                {
                    "parameter": {"min_value": val,
                                "shift": val,
                                "num_params": val}
                }
        scoring: str = 'r2'
            scoring type for the model and its evaluation
        coeff_thresh: float = 1e-8
            Threshold to filter coefficients from trained models
        n_models: int = 1
        n_jobs: int = 1
            number of jobs to train model
        """
        # Create logger
        self.log = self.getLogger(level=level)

        # Parameters for grabbing gene expression information
        self.sample_ind = sample_ind # NOTE for GTEx sample_ind is 4
        self.gene_id_ind = gene_id_ind
        self.gene_chrom_ind = gene_chrom_ind
        self.gene_start_ind = gene_start_ind
        self.gene_end_ind = gene_end_ind

        # Load training and predicting parameters
        self.model = model
        self.model_params = model_params
        self.hyperparams = hyperparams
        self.SearchCV = paramsearchfxn
        self.gene_names = gene_names
        self.train_header_file = train_header_file
        self.predict_header_file = predict_header_file
        self.snp_file = snp_file
        self.str_file = str_file
        self.train_expr_file = train_expr_file
        self.predict_expr_file = predict_expr_file
        self.outpath = outpath

        # Parameters for training
        self.param_shift = param_shift
        self.scoring = scoring
        self.n_jobs = n_jobs
        
        # Used to store outputs of our training process
        self.n_models = n_models
        self.coeff_thresh = coeff_thresh
        self.model_types = model_types

        # Store variants for each gene 
        self.snp_variants = {}
        self.snp_var_info = {}
        self.str_variants = {}
        self.variants = {}
        for gene in gene_names:
            # First index are SNP variants, second are STR
            self.variants[gene] = None

        # Store r2 results as a 3d numpy array (genes, number of models trained, r2_types)
        # Store features in a 2d numpy array (genes, r2_types)
        # Store trained models as 3d numpy array (genes, number of models trained, r2_types)
        self.model_r2 = np.zeros((len(self.gene_names), self.n_models, len(self.model_types)), dtype=np.float32)
        self.model_adj_r2 = np.zeros((len(self.gene_names), self.n_models, len(self.model_types)), dtype=np.float32)
        self.all_features = np.zeros((len(self.gene_names), len(self.model_types)), dtype=np.int32)
        self.all_models = np.zeros((len(self.gene_names), self.n_models, len(self.model_types)), dtype=object)

    def getLogger(self, name: str = '', level: str = "ERROR"):
        """
        Retrieve a Logger object

        Parameters
        ----------
        name : str, optional
            The name of the logging object
        level : str, optional
            The level of verbosity for the logger
        """
        # create logger
        logger = logging.getLogger("Gene Prediction" + name)
        logger.setLevel(level)

        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(level)

        # add ch to logger
        logger.addHandler(ch)

        return logger

    def filter_maf(self, gts, threshold, str_allele_filter=False):
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

    def prepare_header_info(self, header_file):
        # Collect samples names and gene names from expression file
        header_data = np.loadtxt(header_file, delimiter="\t", comments="!", dtype=str)
        samples = list(header_data[0][self.sample_ind:])
        # remove information after period on genename
        genes = header_data[1:,self.gene_id_ind]
        split_genes = np.char.split(genes, sep='.') # Creates numpy array of lists that we need to convert
        split_gene_names = np.array([split_genes[i][0] for i in range(split_genes.shape[0])], dtype=object)
        del header_data

        return samples, split_gene_names

    def collect_gene_region(self, header_file, gene_name, dist):
        """
        Grab region spanning gene with a distance upstream and downstream from 
        gene region of length dist.
        eg. gene coordinates are chr10:1000000-1010000 and dist is 200kb
            region = chr10:800000-1210000

        Parameters
        ----------
        header_file: str
            File to collect gene names and chromosomal location information from
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
        # We want to include the header so we can skip since files will differ in front character
        header_data = np.loadtxt(header_file, delimiter="\t", skiprows=1, usecols=np.arange(self.sample_ind), comments="!", dtype=str)
        # remove information after period on genename
        gene_name = gene_name.split('.')[0]
        split_genes = np.char.split(header_data[:,self.gene_id_ind], sep='.') # Creates numpy array of lists that we need to convert
        split_gene_names = np.array([split_genes[i][0] for i in range(split_genes.shape[0])], dtype=object)
        gene_data = header_data[split_gene_names == gene_name].flatten()
        chrom = gene_data[self.gene_chrom_ind]
        cis_dist = dist*1000
        region = f"{chrom}:{max(1, int(gene_data[self.gene_start_ind]) - cis_dist)}-{int(gene_data[self.gene_end_ind]) + cis_dist}" 
        return chrom, region

    def prepare_gene_data(
        self,
        snp_file,
        str_file,
        samples=None,
        gene_name=None,
        genes=None,
        expr_file=None,
        region=None,
        snp_variants=None,
        str_variants=None,
        filter_freq=True,
        het_thresh=0.1,
        convert_missing=True,
    ):
        # NOTE gene expression data normalized, filtered, and PEER factor residuals removed
        gene_expr_data = None
        if expr_file is not None:
            expr_data = np.loadtxt(expr_file, delimiter=" ", dtype=np.float32)
            gene_expr_data = expr_data[:, genes == gene_name].flatten()

        ### Load SNP Genotypes ###
        # Load Matrices of data, gts located in snp_gts.data and str_gts.data
        # Filter out rare variants
        snp_gts = GenotypesVCF(snp_file)
        snp_gts.read(samples=samples, variants=snp_variants, region=region)

        # Convert all missing data to homozygous reference and output how many samples and alleles were missing
        if convert_missing:
            missing_alleles_per_sample = np.sum(np.sum(snp_gts.data[:,:,:2] > 100, axis=1), axis=1)
            samples_missing = np.sum(missing_alleles_per_sample > 0)
            self.log.info(f"Total Samples Missing Alleles: {samples_missing}")
            self.log.info(f"Total Missing SNP Alleles: {np.sum(missing_alleles_per_sample)}")
            snp_gts.data[snp_gts.data > 100] = 0

        #snp_gts = GenotypesVCF.load(snp_file, samples=samples, variants=snp_variants, region=region)
        snp_gts.subset(samples=samples, inplace=True)

        # Filter those with allele freq < 0.05 (~het < 0.1) 
        if filter_freq:
            snp_gts = self.filter_maf(snp_gts, het_thresh)

        ### Load STR Genotypes ###
        str_gts = GenotypesTR(str_file, vcftype='hipstr')

        # Determine how to filter genotypes
        invalid_ids = (str_variants is not None)
        invalid_ids = invalid_ids or np.any(str_gts.variants["id"] == None)

        if invalid_ids:
            str_gts.read(region=region, samples=samples)
            filtered_vars = np.zeros((len(str_gts.variants["pos"])), dtype=np.bool_)
            # Filter out any variants not in set of wanted variants
            for var_ind, var in enumerate(str_gts.variants["pos"]):
                if var in str_variants["pos"]:
                    filtered_vars[var_ind] = True
            str_gts.data = str_gts.data[:, filtered_vars, :]
            str_gts.variants = str_gts.variants[filtered_vars]

        else:
            str_gts.read(region=region, samples=samples, variants=str_variants)

        # Filter repeated genotypes (the repeats are the same TR just from different tools so just keep the first one)
        unique_vars, unique_indices = np.unique(str_gts.variants["pos"], return_index=True)
        str_gts.data = str_gts.data[:, unique_indices, :]
        str_gts.variants = str_gts.variants[unique_indices]

        # Filter on frequency
        if filter_freq:
            str_gts = self.filter_maf(str_gts, het_thresh, str_allele_filter=True)

        str_gts.subset(samples=samples, inplace=True) # Reorder samples of GTs to the variable samples
        snp_samples, str_samples = GetOverlappingSamples(snp_gts, str_gts)

        # load correct samples and merge for snp_str dataset
        snp_gts.data = snp_gts.data[snp_samples, :, :]
        snp_gts.samples = tuple(np.array(snp_gts.samples)[snp_samples])
        str_gts.data = str_gts.data[str_samples, :, :]
        str_gts.samples = tuple(np.array(str_gts.samples)[str_samples])
        return snp_gts, str_gts, gene_expr_data
    
    def coeff_analysis(self, gene, gene_ind):
        avg_coeff_size = np.zeros((len(self.model_types), len(self.coeff_thresh)))
        # Find best model for current gene and collect coeff
        for model_ind in range(len(self.model_types)):
            self.log.info(f"Calculating avg number of {self.model_types[model_ind]} coeffs")
            for model_num in range(self.all_models.shape[1]):
                # Grab coeffs from model
                best_coef = self.all_models[gene_ind, model_num, model_ind].coef_
                for thresh_ind, thresh_coeff in enumerate(self.coeff_thresh):

                    coef_filter, var_coef_filter, num_nonzero_coeff = self.nonzero_coeff_filter(
                        best_coef, gene, thresh_coeff
                    )
                    avg_coeff_size[model_ind, thresh_ind] += num_nonzero_coeff

                    self.log.info(f"{num_nonzero_coeff} variants contributing to model with threshold of {thresh_coeff}")
                    self.log.debug("Chrom ID Coeff")
                    for var, coeff in zip(self.variants[gene][var_coef_filter], best_coef[coef_filter]):
                        self.log.debug(f"{var[1]} {var[0]} {var[2]} {coeff}")
                    
            avg_coeff_size[model_ind, :] = avg_coeff_size[model_ind, :]/self.all_models.shape[1]
        return avg_coeff_size

    def nonzero_coeff_filter(self, best_coef, gene, thresh_coeff):
        # Grab nonzero coefficients from best model
        coef_filter = (best_coef > thresh_coeff) | (best_coef < -1*thresh_coeff)
        var_coef_filter = coef_filter
        num_nonzero_coeff = np.sum(coef_filter)

        if coef_filter.shape[0] < self.variants[gene].shape[0]:
            var_coef_filter = np.concatenate((coef_filter, np.zeros(self.variants[gene].shape[0] - coef_filter.shape[0])))
            var_coef_filter = np.array(var_coef_filter, dtype=np.bool_)

        return coef_filter, var_coef_filter, num_nonzero_coeff

    def plot_residuals(self, predicted, residual, gene_name, gts_type, outpath):
        title = f"{gene_name} Predicted vs Observed Residuals"
        outfig= outpath + "_".join([gts_type, gene_name, "residuals_plot.png"])
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_title(title)
        ax.set_xlabel("Predicted Gene Expression")
        ax.set_ylabel("Observed Gene Expression")
        ax.scatter(predicted, residual)
        fig.savefig(outfig)
    
    def _repeated_output(self, output):
        """
        Run the training process repeat_num amount of times and output average 
        values with confidence intervals.
        repeat_num: int
            Number of times to repeat the training process
        """
        # Write header to file
        with open(output, 'w') as outfile:
            header = ["Gene", "Num_models_trained"]
            header.extend(
                [
                 "SNP_R_sq_Mean", "SNP_STR_R_sq_Mean", "nonlinear_R_sq_Mean",
                 "SNP_R_sq_Std_Dev", "SNP_STR_R_sq_Std_Dev", "nonlinear_R_sq_Std_Dev",
                 "SNP_adj_R_sq_Mean", "SNP_STR_adj_R_sq_Mean", "nonlinear_adj_R_sq_Mean",
                 "SNP_adj_R_sq_Std_Dev","SNP_STR_adj_R_sq_Std_Dev",  "nonlinear_adj_R_sq_Std_Dev"
                ]
            ) # Should be r2 means, r2 std dev, adj r2 means, adj r2 devs
            
            # extend with coeff threshold from list
            for thresh in self.coeff_thresh:
                header.extend([f"SNP_avg_coeff_{thresh}", 
                               f"SNP_STR_avg_coeff_{thresh}", 
                               f"nonlinear_avg_coeff_{thresh}"])
            header[-1] = header[-1] + "\n"
            outfile.write("\t".join(header))

            # Calc means of adjusted r2 resulting in (genes, model_types = 3)
            adj_r2_means = np.mean(self.model_adj_r2, axis=1)
            adj_r2_stddev = np.std(self.model_adj_r2, axis=1)
            r2_means = np.mean(self.model_r2, axis=1)
            r2_stddev = np.std(self.model_r2, axis=1)

            # Calculate model stats
            for gene_ind, gene in enumerate(self.gene_names):
                # Collect average number of coeff for each model_type
                model_coeffs = self.coeff_analysis(gene, gene_ind)
                self.log.info(f"Each model types' average number of coefficients for thresholds {self.coeff_thresh}:")
                for model_type in range(model_coeffs.shape[0]):
                    self.log.info(f"{self.model_types[model_type]} {model_coeffs[model_type,:]}")

                # Write to file
                out = [f"{gene}", f"{self.all_models.shape[1]}"]
                out.extend([str(r2_mean) for r2_mean in r2_means[gene_ind, :]])
                out.extend([str(r2_std) for r2_std in r2_stddev[gene_ind, :]])
                out.extend([str(adj_r2_mean) for adj_r2_mean in adj_r2_means[gene_ind, :]])
                out.extend([str(adj_r2_std) for adj_r2_std in adj_r2_stddev[gene_ind, :]])
                out.extend([str(coeff) for coeff in model_coeffs.flatten('F')])
                outfile.write("\t".join(out) + "\n")
        return

    def save_best_models(self):
        # save best models
        # best_models has shape genes x model types
        for gene_ind, gene in enumerate(self.gene_names):
            for model_ind, model_type in enumerate(self.model_types):
                best_model = np.argmax(self.model_adj_r2[gene_ind, :, model_ind])
                model_file = self.outpath + f"{gene}_{model_type}.pkl"
                with open(model_file, 'wb') as f:
                    pickle.dump(self.all_models[gene_ind, best_model, model_ind], f)
        return

    def load_best_models(self, gene_name, model_file_schema):
        # load models
        models = []
        for model_type in ["SNPs", "SNPs+STRs", "SNPs+STRs+NonlinearSTRs"]:
            model_file = model_file_schema.format(gene_name=gene_name, model_type=model_type)
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
                models.append(model)
        return models

    def optimize_hyperparameters(
        self,
        X,
        Y,
        model_params,
        hyperparameters,
        title=None,
        cv_figure=None,
    ):
        # Use train test split to evaluate how well it works
        if cv_figure:
            # For figure just do 1 pass to plot and show hyperparameters on train and test data to ensure the best is being chosen
            # NOTE this assumes a single parameter to iterate over
            cv = RepeatedKFold(n_splits=3, n_repeats=2)
            hyperparam = list(hyperparameters.keys())[0]
            hyperparams = hyperparameters[hyperparam]
            train_r2 = []
            test_r2 = []
            all_models = []

            for param in hyperparams:
                # CV for each param and store r2
                cv_model = self.model(**{hyperparam: param}, **model_params)
                results = cross_validate(
                    cv_model, X, Y, scoring="r2", cv=cv, 
                    return_train_score = True
                )
                all_models.append(cv_model)
                train_r2.append(np.mean(results['train_score']))
                test_r2.append(np.mean(results['test_score']))

            best_model = all_models[np.argmax(test_r2)]
            best_r2 = np.amax(test_r2)
            best_model.fit(X,Y)
            self.log.info(f"Best {hyperparam} = {hyperparams[np.argmax(test_r2)]}")

            # Plot train and test r2 to show how scores level out
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.set_title(title)
            ax.set_xlabel(hyperparam)
            ax.set_ylabel("R Squared")
            ax.plot(hyperparams, train_r2, label="Train")
            ax.plot(hyperparams, test_r2, label="Test")
            ax.legend()
            fig.savefig(cv_figure)

        else:
            # First more expansive search for best hyperparameter
            cv = RepeatedKFold(n_splits=3, n_repeats=2)
            first_search = self.SearchCV(
                self.model(**model_params),
                hyperparameters,
                scoring=self.scoring,
                cv=cv,
                n_jobs=self.n_jobs).fit(X, Y)
            params = first_search.best_params_

            if not self.param_shift:
                return first_search.best_estimator_, first_search.best_score_

            for key in params.keys():
                min_value = self.param_shift[key]["min_value"]
                shift = self.param_shift[key]["shift"]
                num_params  = self.param_shift[key]["num_params"]
                params[key] = np.linspace(max(min_value, params[key]-shift), params[key]+shift, num_params)

            # Second and final search more fine tuned for best hyperparameter
            results = self.SearchCV(self.model(**model_params), params, scoring='r2', cv=cv).fit(X, Y)
            best_model = results.best_estimator_
            best_r2 = results.best_score_
            self.log.info(f"Best parameters = {results.best_params_}")

        return best_model, best_r2
    
    def evaluate_models(self, models, gt_data, true_expr, gene_name, gt_scaler_schema):
        r_squared = []
        adj_r_squared = []
        for model_ind, model in enumerate(models):
            # predict expression
            gt_scaler_file = gt_scaler_schema.format(gene_name, self.model_types[model_ind].replace('+', '_').lower())
            with open(gt_scaler_file, 'rb') as gt_f:
                gt_scaler = pickle.load(gt_f)
            X = gt_scaler.transform(gt_data[model_ind])

            # If the model is nonlinear need to take abs value 
            if model_ind == len(models) - 1:
                X[:, gt_data[model_ind-1].shape[1]:] = np.abs(X[:, gt_data[model_ind-1].shape[1]:])

            pred_expr = model.predict(X)

            # Compute adjusted r2
            r2 = np.corrcoef(true_expr, pred_expr)[0, 1] ** 2
            n = X.shape[0] # number of samples
            coef = model.coef_
            coef_filter, var_coef_filter, p = self.nonzero_coeff_filter(
                coef, gene_name, 0.01
            )
            adj_r2 = 1 - ((1 - r2)*(n-1)/(n-p-1))
            r_squared.append(r2)
            adj_r_squared.append(adj_r2)
        return r_squared, adj_r_squared

    def predict_models(
        self, 
        model_file_schema,
        gt_scaler_schema,
        coeff_plots=False
    ):
        # Grab sample names and genes from header info like we do below
        samples, header_genes = self.prepare_header_info(self.predict_header_file)

        # Create output file
        output_file = self.outpath + f"AFR_{self.gene_names[0]}_prediction_results.txt"
        with open(output_file, 'w') as output:
            # Write output header
            output.write(f"Gene\tSNP_R_sq\tSNP_STR_R_sq\tnonlinear_R_sq\tSNP_adj_R_sq\tSNP_STR_adj_R_sq\tnonlinear_adj_R_sq\n")

            # Load genotypes and gene expression data
            for gene_ind, gene_name in enumerate(self.gene_names):
                # Collect the SNP variants used from train header file 
                #    since these are the variants used in the trained model
                header_data = np.loadtxt(
                    self.train_header_file,
                    delimiter="\t",
                    dtype=str,
                )
                eur_samples = list(header_data[0][self.sample_ind:])
                train_dataset, train_expr = self.collect_gts_gene_expr(
                    gene_name,
                    500,
                    header_genes,
                    eur_samples,
                    snp_variants=None,
                    str_variants=None,
                    expr_file=self.train_expr_file,
                    header_file=self.train_header_file,
                )

                # Collect Gts and expression to use from samples we want to predict on
                predict_dataset, predict_expr = self.collect_gts_gene_expr(
                    gene_name,
                    500,
                    header_genes,
                    samples,
                    snp_variants=self.snp_variants[gene_name],
                    str_variants=self.str_variants[gene_name],
                    expr_file=self.predict_expr_file,
                    header_file=self.predict_header_file,
                    filter_freq=False,
                )

                # Load prior trained models
                models = self.load_best_models(gene_name, model_file_schema)

                # Collect R-squared values for each model type and output
                models_r_squared, models_adj_r_squared = self.evaluate_models(
                    models, 
                    predict_dataset, 
                    predict_expr, 
                    gene_name,
                    gt_scaler_schema,
                )

                if coeff_plots:
                    # Make directory of .../Gene_name/
                    # save all plots in said directory
                    # Load given gene expression data and complimentary genotypes for the give gene
                    self.gene_expr_plots(
                        models[1],
                        gene_name,
                        train_expr,
                        predict_expr,
                        train_dataset[1],
                        predict_dataset[1],
                    )

                # Save r-squared results to output
                output.write(
                    f"{gene_name}\t{models_r_squared[0]}\t{models_r_squared[1]}\t{models_r_squared[2]}\t"
                    f"{models_adj_r_squared[0]}\t{models_adj_r_squared[1]}\t{models_adj_r_squared[2]}\n"
                )

        return 
    
    def predict_expression(
        self, 
        models,
        gt_scalers,
        expr_scaler,
        feature_snp_variants,
        feature_tr_variants,
        samples,
        outpath
    ):
        """
        Steps for predicting gene expression in VCF samples
            1. Load Genotypes and only keep variants from set of features used to train
                i) Variants not present from list of features will be given all 0s
            2. Predict Expression for each sample
            3. Output Expression
        """

        # Load genotypes and gene expression data
        for gene_ind, gene_name in enumerate(self.gene_names):
            #### 1. Load Genotypes and only keep variants from list fo features ####

            # load gene region
            chrom, gene_region = self.collect_gene_region(self.predict_header_file, gene_name, 500)

            snp_file = self.snp_file.format(chrom)
            str_file = self.str_file.format(chrom)

            # Load genotypes
            snp_gts, str_gts, gene_expr_data = self.prepare_gene_data(
                snp_file,
                str_file,
                samples=list(samples),
                region=gene_region,
                filter_freq=False,
                convert_missing=False,
            )

            # Load start position and ref alt for snp and start pos for trs (no duplicates in TRs)
            loaded_snp_coords = np.array(snp_gts.variants.tolist(), dtype=object)[:,2].astype(np.uint32)
            loaded_snp_ref_alt = np.array(snp_gts.variants.tolist(), dtype=object)[:,3:]
            loaded_tr_coords = np.array(str_gts.variants.tolist())[:,2].astype(np.uint32)

            # Samples x Variants
            snp_gts.data = snp_gts.data[:,:,:2] # Remove phasing flag
            str_gts.data = str_gts.data[:,:,:2] # Remove phasing flag
            str_gts = np.array(str_gts.data, dtype=np.int32)

            # Filter to only use model variants
            model_gts = np.zeros(
                (snp_gts.data.shape[0], feature_snp_variants.shape[0] + feature_tr_variants.shape[0])
            )
            
            # Create dict to match variants in model (Pos, ref, alt)
            feature_snp_dict = {}
            for snp_ind, snp_data in enumerate(feature_snp_variants):
                pos = snp_data[0]
                ref = snp_data[1]
                alt = snp_data[2]

                if feature_snp_dict.get(pos) is None:
                    feature_snp_dict[pos] = []
                feature_snp_dict[pos].append((snp_ind, ref, alt))

            feature_matching_inds = []
            loaded_matching_inds = []
            for load_snp_ind in range(loaded_snp_coords.shape[0]):
                grabbed_alleles = feature_snp_dict.get(loaded_snp_coords[load_snp_ind])
                
                if grabbed_alleles is None:
                    continue
                
                # Iterate over grabbed alleles and match ref alt then grab indices where it matches
                for allele in grabbed_alleles:
                    loaded_alleles = loaded_snp_ref_alt[load_snp_ind][0]
                    if (allele[1] == loaded_alleles[0]) and (allele[2] == loaded_alleles[1]):
                        feature_matching_inds.append(allele[0])
                        loaded_matching_inds.append(load_snp_ind)

            # Convert missing alleles to homozygous reference
            snp_gts = snp_gts.data[:,loaded_matching_inds,:]
            missing_alleles_per_sample = np.sum(np.sum(snp_gts > 100, axis=1), axis=1)
            samples_missing = np.sum(missing_alleles_per_sample > 0)
            self.log.info(f"Total Samples Missing Alleles: {samples_missing}")
            self.log.info(f"Total Missing SNP Alleles: {np.sum(missing_alleles_per_sample)}")
            snp_gts[snp_gts > 100] = 0

            # Sum alleles to make genotypes
            snp_gts = np.sum(snp_gts, axis=2)
            nonlin_gts = np.sum(np.square(str_gts), axis=2)
            str_gts = np.sum(str_gts, axis=2)

            # Load matching snp_gts 
            model_gts[:,feature_matching_inds] = snp_gts

            # Check to make sure all positions are the same
            #for feat_pos, loaded_pos in zip(feature_snp_variants['pos'][feature_matching_inds], loaded_snp_coords[loaded_matching_inds]):
            #    assert feat_pos == loaded_pos
            
            # Use this for TRs make a different workflow for SNPs since it has pos, ref, alt to match
            def matching_variant_mask(matching_variant_array, variant_array_to_mask):
                # find overlapping indices of the model feature variants and genotypes loaded from file
                matching_inds = np.searchsorted(matching_variant_array, variant_array_to_mask)
                matching_inds[matching_inds == matching_variant_array.shape[0]] = matching_variant_array.shape[0] - 1 # update indices that are = size of the array
                mask = variant_array_to_mask == matching_variant_array[matching_inds]
                return mask

            tr_loaded_gt_mask = matching_variant_mask(feature_tr_variants, loaded_tr_coords)
            tr_model_gt_mask = matching_variant_mask(loaded_tr_coords, feature_tr_variants)
            model_gt_mask = np.append(np.zeros((feature_snp_variants.shape[0],)), tr_model_gt_mask).astype(bool)            
            
            # Load TR gts into array to be passed into model
            model_gts[:,model_gt_mask] = str_gts[:,tr_loaded_gt_mask]
            model_gts = np.concatenate((model_gts, np.zeros((model_gts.shape[0], feature_tr_variants.shape[0]))), axis=1)
            model_nonlin_gt_mask = np.append(np.zeros((feature_snp_variants.shape[0]+feature_tr_variants.shape[0],)), tr_model_gt_mask).astype(bool)
            model_gts[:,model_nonlin_gt_mask] = nonlin_gts[:,tr_loaded_gt_mask]

            #### 2. Predict Expression for each sample ####

            #### Inverse transform the predicted data to get the unscaled predicted gene expression values ####
            pred_expr = []
            for model, scaler, sample_gts in zip(models, gt_scalers, [model_gts[:,: feature_snp_variants.shape[0]], model_gts[:,:feature_snp_variants.shape[0]+feature_tr_variants.shape[0]], model_gts]):
                scaled_gts = scaler.transform(sample_gts)
                # Take absolute value for nonlinear gts to make sure they hold their shape and cant be negative
                if sample_gts.shape == model_gts.shape:
                    scaled_gts[:,feature_snp_variants.shape[0]+feature_tr_variants.shape[0]:] = np.abs(scaled_gts[:,feature_snp_variants.shape[0]+feature_tr_variants.shape[0]:])
                pred_expr.append(
                    expr_scaler.inverse_transform(
                        model.predict(scaled_gts).reshape(-1,1)
                    )
                )

            #### 3. Output Expression ####
            for final_expr, model_type in zip(pred_expr, ["snp", "snp_tr", "nonlinear"]):
                np.savetxt(outpath + f"{gene_name}_{model_type}_predicted_expr.csv", final_expr, delimiter='\t')

        return

    def output_results(self):
        outfile = self.outpath + f"{self.gene_names[0]}_evaluation_metrics.txt"
        self._repeated_output(outfile)
        self.save_best_models()
        return

    def gene_expr_plots(
        self, 
        snp_str_model, 
        gene_name, 
        EUR_expr, 
        AFR_expr,
        EUR_gts,
        AFR_gts,
    ):
        # Check if directory self.outpath + gene_name/ exists
        outdir = self.outpath + gene_name + '/'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        # Use models to grab what variants we want
        coeff = snp_str_model.coef_
        coef_filter, var_coef_filter, num_nonzero_coeff = self.nonzero_coeff_filter(
            coeff, gene_name, 0.01
        )

        # Grab all nonzero variants
        wanted_vars = len(self.snp_variants[gene_name]) + len(self.str_variants[gene_name])
        var_coef_filter = var_coef_filter[:wanted_vars]
        snp_str_vars = self.variants[gene_name][:wanted_vars]
        nonzero_vars = snp_str_vars[var_coef_filter]

        # Load EUR and AFR Genotypes
        EUR_nonzero_gts = EUR_gts[:, var_coef_filter]
        AFR_nonzero_gts = AFR_gts[:, var_coef_filter]

        # Plot European and AFR on top of each other (Have like a 0.1 or 0.2 separation so we can see the dots distinctly)
        for variant in range(EUR_nonzero_gts.shape[1]):
            self.plot_variant_gene_expr(
                nonzero_vars[variant],
                gene_name,
                EUR_nonzero_gts[:,variant], 
                AFR_nonzero_gts[:,variant], 
                EUR_expr, 
                AFR_expr,
                outdir,
            )
        return

    def plot_variant_gene_expr(
        self, 
        var_info, 
        gene_name, 
        EUR_var, 
        AFR_var, 
        EUR_expr, 
        AFR_expr,
        outdir,
    ):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        variant_full_info = f"{var_info[0]}".replace(':', '_')
        if var_info[0] == 'None':
            variant_full_info = f"Repeat_{var_info[1]}_{var_info[2]}"
        
        ax.set_title(f"{variant_full_info} vs {gene_name}")
        ax.set_xlabel("Genotypes (A1+A2)")
        ax.set_ylabel("Expression")
        ax.scatter(EUR_var, EUR_expr, color='b', alpha=0.5, label="EUR")
        ax.scatter(AFR_var + 0.2, AFR_expr, color='r', alpha=0.5, label="AFR")
        ax.legend()
        fig.savefig(outdir + f"{gene_name}_{variant_full_info}.png")

        return
    
    def collect_gts_gene_expr(
        self, 
        gene_name, 
        gene_dist, 
        header_genes, 
        samples,
        snp_variants,
        str_variants,
        expr_file,
        header_file,
        filter_freq=True,
    ):
        chrom, gene_region = self.collect_gene_region(header_file, gene_name, gene_dist)

        snp_file = self.snp_file.format(chrom)
        str_file = self.str_file.format(chrom)

        snp_gts, str_gts, gene_expr_data = self.prepare_gene_data(
            snp_file,
            str_file,
            samples,
            gene_name,
            header_genes,
            expr_file,
            region=gene_region,
            snp_variants=snp_variants,
            str_variants=str_variants,
            filter_freq=filter_freq,
        )
        # Save for usage in collecting information about which variants are used
        self.snp_variants[gene_name] = snp_gts.variants["id"]
        self.snp_var_info[gene_name] = snp_gts.variants
        self.str_variants[gene_name] = str_gts.variants

        # Assert that the variants are identical if snp and str_variants are given
        if snp_variants is not None:
            assert snp_variants.shape[0] == snp_gts.variants.shape[0]
            np.testing.assert_array_equal(snp_variants, snp_gts.variants["id"])
        if str_variants is not None:
            np.testing.assert_array_equal(str_variants["chrom"], str_gts.variants["chrom"])
            np.testing.assert_array_equal(str_variants["pos"], str_gts.variants["pos"])

        snp_str_gts = Genotypes.merge_variants((snp_gts, str_gts), fname=None)
        self.variants[gene_name] = snp_str_gts.variants
        self.variants[gene_name] = np.concatenate((self.variants[gene_name], str_gts.variants))
        snp_str_gts.data = snp_str_gts.data[:,:,:2] # Remove phasing flag
        snp_str_gts = snp_str_gts.data
        
        # Increase datatype (int32) for squaring of strs
        # Sum GTs across axes (In squared case, square first then sum)
        snp_gts.data = snp_gts.data[:,:,:2] # Remove phasing flag
        str_gts.data = str_gts.data[:,:,:2] # Remove phasing flag
        snp_gts = np.sum(snp_gts.data, axis=2)
        str_gts = np.array(str_gts.data, dtype=np.int32)

        # Collect means for nonlinear shift 
        converted_str_gts = np.transpose(str_gts, (2,0,1))
        str_all_alleles = np.reshape(converted_str_gts,(str_gts.shape[0]*str_gts.shape[2], str_gts.shape[1])) # reshape so pairs of alleles become one col

        # Get most common alleles for each variant to use for replacing non-genotyped individuals
        allele_modes, counts = scipy.stats.mode(str_all_alleles, axis=0)

        # replace all non present TR alleles with the most common allele
        str_all_alleles = np.where(str_all_alleles >= 254, allele_modes, str_all_alleles)

        # Take mean of TR alleles
        #str_means = np.mean(str_all_alleles, axis=0)[:, np.newaxis]
        
        # Duplicate means and modes so theyll work on the 3d array
        #str_means = np.repeat(str_means, repeats=2, axis=1)
        allele_modes = np.repeat(allele_modes[:,np.newaxis], repeats=2, axis=1)
        #allele_modes = allele_modes.transpose(0,2,1)

        # Filter out non present TR alleles and replace with most common allele
        merged_tr_gts = snp_str_gts[:,snp_gts.shape[1]:,:]
        merged_tr_gts = np.where(merged_tr_gts >= 254, allele_modes, merged_tr_gts)
        nonlin_str_gts = merged_tr_gts

        # Collect SNPs, mean subtracted TRs, and mean subtracted nonlinear TRs
        nonlin_gts = np.sum(np.square(nonlin_str_gts), axis=2)
        nonlin_str_gts = np.sum(nonlin_str_gts, axis=2)
        nonlin_snp_gts = np.sum(snp_str_gts[:,:snp_gts.shape[1],:], axis=2)
        snp_str_gts = np.concatenate((nonlin_snp_gts, nonlin_str_gts), axis=1)
        nonlinear_snp_str_gts = np.concatenate((snp_str_gts, nonlin_gts), axis=1)

        # iterate over each model type we want to train
        datasets = [snp_gts, snp_str_gts, nonlinear_snp_str_gts]
        return datasets, gene_expr_data
    
    def train_model(
        self,
        X,
        Y,
        gene_name=None,
        model_params=None,
        hyperparameters=None,
        title=None,
        cv_figure=None,
    ):
        """
        X: np.array()
            Predictor variable. X must be a numpy array in the shape of samples x features.
            X should already be scaled.
        Y: np.array()
            Predicted variable. Y must be a numpy array in the shape of samples x 1.
            Y should already be scaled.
        """
        self.log.info(f"Dataset has {X.shape[0]} samples and {X.shape[1]} features")

        # Check if we need to do Cross Validation or best param already given
        for key in hyperparameters:
            if isinstance(hyperparameters[key], Iterable) and len(hyperparameters[key]) > 1:
                self.log.info("Optimizing hyperparameters.")
                # For CV perform two sets of CV to narrow down param as much as possible
                final_model, test_r2 = self.optimize_hyperparameters(
                    X,
                    Y,
                    model_params,
                    hyperparameters,
                    title=title, 
                    cv_figure=cv_figure)
                return final_model, test_r2

        # Know hyperparameter so use CV to calc test score and final model
        cv = RepeatedKFold(n_splits=3, n_repeats=2)
        final_model = self.model(**hyperparameters, **model_params)
        results = cross_validate(
            final_model, X, Y, scoring="r2", cv=cv, 
        )
        test_r2 = np.mean(results['test_score'])
        final_model.fit(X, Y)

        return final_model, test_r2
    
    def iterate_train(
        self,
        gts,
        gts_type,
        gene_name,
        gene_expr_data,
        model_params,
        hyperparams,
        gene_ind,
        model_ind,
        outpath,
    ):
        for num_model in range(self.n_models):
            output_fig = None
            figure_title = None
            if outpath:
                output_fig = outpath + "_".join([gts_type, gene_name, "hyperparameter_convergence.png"])
                figure_title = f"{gene_name} Hyperparam vs R Squared"
            self.log.info(f"Training model {gts_type} {num_model}")

            # load up separate models (snp only and snp_str)
            final_model, r2 = self.train_model(
                gts,
                gene_expr_data,
                gene_name=gene_name,
                model_params=model_params,
                hyperparameters=hyperparams,
                title=figure_title,
                cv_figure=output_fig,
            )

            self.model_r2[gene_ind, num_model, model_ind] = r2
            self.all_models[gene_ind, num_model, model_ind] = final_model

            # Compute adjusted r2
            n = gts.shape[0]
            coef = self.all_models[gene_ind, num_model, model_ind].coef_
            coef_filter, var_coef_filter, p = self.nonzero_coeff_filter(
                coef, gene_name, 0
            )
            adj_r2 = 1 - ((1 - r2)*(n-1)/(n-p-1))
            self.model_adj_r2[gene_ind, num_model, model_ind] = adj_r2
            self.log.info(f"{gts_type} model test predicted r2: {r2} and adjusted r2: {adj_r2}")
        return

    def run_model_pipeline(
        self,
        gene_dist=500,
        snp_variants=None,
        str_variants=None,
        outpath=None,
    ):
        eur_samples, header_genes = self.prepare_header_info(self.train_header_file)

        for gene_ind, gene_name in enumerate(self.gene_names):
            datasets, gene_expr_data = self.collect_gts_gene_expr(
                gene_name, 
                gene_dist, 
                header_genes, 
                eur_samples,
                snp_variants,
                str_variants,
                expr_file=self.train_expr_file,
                header_file=self.train_header_file,
            )

            # TODO uncomment for outputting variant info currently commented for testing nonlinear 
            # output the variants used to train the model
            #   For snps use snp_var_info
            #   For trs use str_variants
            #model_vars_output = self.outpath + f"{gene_name}_trained_model_variant_info.txt"
            #with open(model_vars_output, 'w') as var_out:
            #    var_out.write('VarType\tChrom\tStart\tRef\tAlt\n')
            #    for snp_var in self.snp_var_info[gene_name]:
            #        var_out.write(f'SNP\t{snp_var[1]}\t{snp_var[2]}\t{snp_var[3][0]}\t{snp_var[3][1]}\n')
            
            #    for tr_var in self.str_variants[gene_name]:
            #        var_out.write(f'TR\t{tr_var[1]}\t{tr_var[2]}\n')

            # Normalize Gene Expression data
            Y_scaler = StandardScaler()
            Y = gene_expr_data.reshape(-1, 1)
            Y = Y_scaler.fit_transform(Y)

            # If Only one model is given for model_types only use that dataset
            filtered_datasets = []
            all_model_types = np.array(["SNPs", "SNPs+STRs", "SNPs+STRs+NonlinearSTRs"]).astype(object)
            dataset_mask = np.where(self.model_types == all_model_types)[0]
            for data_ind in dataset_mask:
                filtered_datasets.append(datasets[data_ind])

            # Save Gene Expression Normalization
            with open(self.outpath + f'{gene_name}_expr_scaler.pkl', 'wb') as y_file:
                pickle.dump(Y_scaler, y_file)

            for model_ind, (gts, gts_type) in enumerate(zip(filtered_datasets, self.model_types)):
                # Load parameters
                model_params = self.model_params[gts_type]
                hyperparams = self.hyperparams[gts_type]
                # Store feature counts
                self.all_features[gene_ind][model_ind] = gts.shape[1]

                # standardize all data
                X_scaler = StandardScaler()
                X = X_scaler.fit_transform(gts)

                # Output standard scalers for future data prediction
                gts_type_path = gts_type.replace('+', '_').lower()

                with open(self.outpath + f'{gene_name}_{gts_type_path}_gt_scaler.pkl', 'wb') as x_file:
                    pickle.dump(X_scaler, x_file)

                self.iterate_train(
                    X,
                    gts_type,
                    gene_name,
                    Y,
                    model_params,
                    hyperparams,
                    gene_ind,
                    model_ind,
                    outpath,
                )
        return

# Look at 500kb upstream and downstream to get close to 1Mb total that Gusev paper did
# subsetted snp data from /gymreklab-tscc/yal084_storage/ensemble_phasing/SNP_chr21.vcf.gz and /gymreklab-tscc/yal084_storage/ensemble_phasing/SNP_chr11.vcf.gz
# subsetted str data from /gymreklab-tscc/helia/ensembl/ensemble_out/before_revision_files/ensemble_chr21_filtered.vcf.gz and /gymreklab-tscc/helia/ensembl/ensemble_out/ensemble_chr11_filtered.vcf.gz
# EUR normalized expression data from /expanse/protected/gymreklab-dbgap/mount/yal084/Geuvadis_data/results/EUR_normalized_and_filtered_hg38.csv
# NOTE these files are preloaded with snps and strs upstream and downstream data of 1Mb
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_header_file', type=str, help="Header file to collect gene location information and sample names.")
    parser.add_argument('--train_phenotype_file', type=str, help="Phenotype file where rows are samples and columns are phenotypes. Rows of samples in phenotype file must match header file.")
    parser.add_argument('--predict_header_file', type=str, default=None, help="Header file for samples that models are being used on. Same structure as train header files.")
    parser.add_argument('--predict_phenotype_file', type=str, default=None, help="Phenotype file where rows are samples and columns are phenotypes. Must match header file.")
    parser.add_argument('--snp_genotypes_file', type=str, help="SNP VCF file. This can be formatted by chromosome if the files are split by chrom. Ex: --snp_genotypes_file SNP_{}.vcf.gz")
    parser.add_argument('--str_genotypes_file', type=str, help="STR VCF file. This can be formatted by chromosome if the files are split by chrom. Ex: --str_genotypes_file STR_{}.vcf.gz")
    parser.add_argument('--gt_scaler', type=str, help="Store formatted string of path to scaler pickle files. e.g. /path/to/my_scaler_{}_{}.pkl one {} for gene_name and one for model type")
    parser.add_argument('--header_file_indices', action='store', type=int, nargs='+', help="Indices for header file where to collect gene chrom, gene start, gene end, gene id, starting sample. Ex: --header_file_indices 0 1 2 3 5", required=True)
    parser.add_argument('--gene_names', type=str, help='gene name to generate model for ex: CSTB,EFHB')
    parser.add_argument('--outpath', type=str, help='output file path to save model metrics and pickle files')
    parser.add_argument('--observed_expected', type=str, default=None)
    parser.add_argument('--model_type', type=str, choices=["SNPs", "SNPs+STRs", "SNPs+STRs+NonlinearSTRs"], help='Only run training on one model type. Model types: SNPs, SNPs+STRs, SNPs+STRs+NonlinearSTRs')
    parser.add_argument('--repeat', type=int, default=1, help='Optional: Repeat the training process n number of times where n is an inputted integer > 1')
    parser.add_argument('--model_files', type=str, default=None, help='File format to load for loading gene expression models and predicting on genotypes. Ex: {gene_name}_{model_type}.pkl')
    parser.add_argument('--coeff_plots', action='store_true', help='Look into the coeff of each variant and plot the genotypes vs gene expression plots')
    args = parser.parse_args()

    # NOTE these files are SNP_{chrom}.vcf.gz for example SNP_chr21.vcf.gz so we pass these in and use formatted string to fill in {}
    #EUR_header_file = "/storage/mlamkin/workspace/TWAS-str/EUR_normalized_and_filtered_hg38.csv"
    #AFR_header_file = "/storage/mlamkin/workspace/TWAS-str/AFR_normalized_and_filtered_hg38.csv"
    #EUR_expr_file = "./EUR_normalized_filtered_final_residuals_hg38.csv"
    #AFR_expr_file = "./AFR_normalized_filtered_final_residuals_hg38.csv"
    #snp_file = "/gymreklab-tscc/yal084_storage/ensemble_phasing/SNP_{}.vcf.gz"
    #str_file = "/gymreklab-tscc/helia/ensembl/ensemble_out/ensemble_{}_filtered.vcf.gz"
    #EUR_header_file = "/expanse/projects/gymreklab/mlamkin/TWAS-str/EUR_normalized_and_filtered_hg38.csv"
    #AFR_header_file = "/expanse/projects/gymreklab/mlamkin/TWAS-str/AFR_normalized_and_filtered_hg38.csv"
    #EUR_expr_file = "/expanse/projects/gymreklab/mlamkin/TWAS-str/EUR_normalized_filtered_final_residuals_hg38.csv"
    #AFR_expr_file = "/expanse/projects/gymreklab/mlamkin/TWAS-str/AFR_normalized_filtered_final_residuals_hg38.csv"
    #snp_file = "/expanse/projects/gymreklab/mlamkin/TWAS-str/SNP_vcfs/SNP_{}.vcf.gz"
    #str_file = "/expanse/projects/gymreklab/mlamkin/TWAS-str/STR_vcfs/ensemble_{}_filtered.vcf.gz"\
    #sample_ind = 5 # NOTE for GTEx sample_ind is 4
    #gene_id_ind = 3
    #gene_chrom_ind = 0
    #gene_start_ind = 1
    #gene_end_ind = 2
    
    gene_chrom_ind, gene_start_ind, gene_end_ind, gene_id_ind, sample_ind = args.header_file_indices
    outpath = args.outpath
    if outpath[-1] != '/':
        outpath += '/'
    paramsearch = GridSearchCV

    max_iter = 50000
    gene_names = [gene_name.split('.')[0] for gene_name in args.gene_names.split(',')]
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
    coeff_thresh = [0.01, 0.001, 1e-8]

    # Grab what type of model we should be training
    if args.model_type is not None:
        model_types = [args.model_type]
    else:
        model_types = ["SNPs", "SNPs+STRs", "SNPs+STRs+NonlinearSTRs"]

    test_predictor = GeneExpressionPredictor(
        paramsearchfxn=paramsearch,
        model=lasso_model,
        model_params=model_params,
        hyperparams=hyperparameters,
        gene_names=gene_names,
        sample_ind=sample_ind,
        gene_id_ind=gene_id_ind,
        gene_chrom_ind=gene_chrom_ind,
        gene_start_ind=gene_start_ind,
        gene_end_ind=gene_end_ind,
        train_header_file=args.train_header_file,
        predict_header_file=args.predict_header_file,
        snp_file=args.snp_genotypes_file,
        str_file=args.str_genotypes_file,
        train_expr_file=args.train_phenotype_file,
        predict_expr_file=args.predict_phenotype_file,
        param_shift=shift_params,
        coeff_thresh=coeff_thresh,
        outpath=outpath,
        model_types=model_types,
        n_models=args.repeat,
        n_jobs=1,
        level="DEBUG",
    )

    # collect model files
    model_files = args.model_files
    # predict results from AFR files
    if model_files:
        test_predictor.predict_models(
            model_files, 
            args.gt_scaler,
            args.coeff_plots,
        )
    else:
        test_predictor.run_model_pipeline(gene_dist=500)
        test_predictor.output_results()

if __name__ == '__main__':
    main()
