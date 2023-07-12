import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

import functions

np.random.seed(42)

# Parameters used for clustering
BASEDIR = 'data'  # directory of logs
DEST_FOLDER = 'results 2021-05-01 (k, 1-5 ngrams, nmf)'
CLUSTERING_ALG = 'K-Means'  # clustering algorithm: K-Means, DBSCAN
# the parameters are set depending on the clustering algorithm:
# number of clusters if K-Means, maximum distance between two samples for one to be considered as in the
# neighborhood of the other if DBSCAN
# STEP is the step used to select the parameters
if CLUSTERING_ALG == 'K-Means':
    MIN_VALUE = 2
    MAX_VALUE = 10
    STEP = 1
elif CLUSTERING_ALG == 'DBSCAN':
    MIN_VALUE = 0.1
    MAX_VALUE = 1
    STEP = 0.1
# minimum and maximum number of n-grams to be generated to create more features
MIN_NGRAMS = 1
MAX_NGRAMS = 5
EVAL_METHOD = 'Elbow Method'  # clustering evaluation method: Elbow Method, Silhouette Score
DIM_RED_METHOD = 'NMF'  # dimensionality reduction method: PCA, SparsePCA, NMF
# lists of files and field to be analysed
FILES = ['2021-05-01', '2021-06-28', '2022-01-03']
FILES = ['2021-05-01']
targets = ['NBG_EM_SIM_ALARMS', 'NBG_FM_ALM_SIM_SETUP_PSS24X01_tdm_fm_setup_sim_Testcase-1', 'tdm_fm_setup_sim_Testcase-1']
FIELDS = ['profilename', 'testcase', 'profilename_testcase']
FIELDS = ['profilename']

cluster_total_log = False  # if the clustering of the entire log will be done
cluster_single_fields = True  # if the clustering of the single fields will be done
save_variables = True  # used to save the results at the end

# compute plots
plot_clustering_score = False
plot_heatmap = False
plot_variance = False
plot_scatter_matrix = False

# convert a collection of raw commands to a matrix of TF-IDF features
vect = TfidfVectorizer(ngram_range=(MIN_NGRAMS, MAX_NGRAMS))

# the folder that will contain the results is created
if save_variables:
    results_path = 'workspace/{}'.format(DEST_FOLDER)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # the parameters are saved
    filename = 'workspace/{}/parameters.pkl'.format(DEST_FOLDER)
    parameters = [MIN_VALUE,
                  MAX_VALUE,
                  MIN_NGRAMS,
                  MAX_NGRAMS,
                  EVAL_METHOD,
                  CLUSTERING_ALG,
                  STEP,
                  DIM_RED_METHOD]
    functions.save_parameters(parameters, filename)

doc_per_profile = []  # array that contains all the commands in the log

# iterate for every log and for every field
for file in FILES:
    for field in FIELDS:

        # the folder of the plots are created if they do not exist yet
        path = 'plots/{}_{}'.format(field, file)
        if not os.path.exists(path):
            os.makedirs(path)
        if plot_heatmap and not os.path.exists('{}/heatmaps'.format(path)):
            os.makedirs('{}/heatmaps'.format(path))
        if plot_scatter_matrix and not os.path.exists('{}/scatter matrices'.format(path)):
            os.makedirs('{}/scatter matrices'.format(path))
        if plot_clustering_score and not os.path.exists('{}/clustering scores'.format(path)):
            os.makedirs('{}/clustering scores'.format(path))
        if plot_variance and not os.path.exists('{}/variance explained ratio'.format(path)):
            os.makedirs('{}/variance explained ratio'.format(path))
        print('plots folder created')

        # a structured variable already containing the result of parsing is created to avoid reading
        # the log every time
        documents = {}
        filename = 'workspace/documents_{}_{}.pkl'.format(field, file)
        with open(filename, 'rb') as f:
            documents = pickle.load(f)

        # print(documents)
        # prints all the names found in a log depending of the field
        print('{} {}s found:'.format(len(documents), field))
        for profile in documents:
            print(profile)

        # the clustering is done for every document in the log (a document is the command of a profile or the
        # commands of a testcase)
        for document in documents:
            if document in targets:
                document_series = pd.Series(documents[document])
                # the commands are appended to an array, they will be concatenated to form the series
                # containing all the commands in a log
                doc_per_profile.append(document_series)
                if cluster_single_fields:
                    functions.compute_clustering(documents, document, vect, field, file, CLUSTERING_ALG,
                                                 MIN_VALUE, MAX_VALUE, STEP, EVAL_METHOD, DIM_RED_METHOD,
                                                 plot_clustering_score, plot_heatmap, plot_variance,
                                                 plot_scatter_matrix, save_variables, results_path, document_series)

# the clustering for the entire log is the same
if cluster_total_log:
    document = 'TOTAL_LOG'
    # all the Series are concatenated to obtain a Series that contains all the commands
    documents_in_log = pd.concat(doc_per_profile)
    documents[document] = documents_in_log
    functions.compute_clustering(documents, document, vect, field, file, CLUSTERING_ALG,MIN_VALUE,
                                 MAX_VALUE, STEP, EVAL_METHOD, DIM_RED_METHOD, plot_clustering_score,
                                 plot_heatmap, plot_variance, plot_scatter_matrix, save_variables, results_path,
                                 documents_in_log)

print('end')
