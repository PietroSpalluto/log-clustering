import os
import pickle

import pandas as pd
import numpy as np
import re

from kneed import KneeLocator
from matplotlib import pyplot as plt
from numpy import unique
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import NMF, TruncatedSVD, PCA, SparsePCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import seaborn as sns

np.random.seed(42)

# parameters
FORMAT = 'png'
round_precision = 1  # depends on the step of DBSCAN (1 if STEP=0.1, 2 if STEP=0.01, ...)

compute_silhouette_score = False  # False only if the evaluation method is the elbow method


# parsing of the log to replace the values of the parameters with the name of a parameter
def log_parser(tree, cmd_log):

    # sometimes a command ends with a blank space so it is removed
    if cmd_log[-1] == ' ':
        cmd_log = cmd_log[0:-1]

    splitted_command = re.split(r'\s+', cmd_log)
    last_tree_position = tree
    parameter_regex = re.compile(r'<[\w-]+>$')
    log_template = []

    for command_keyword in splitted_command:
        if command_keyword in last_tree_position:
            last_tree_position = last_tree_position[command_keyword]
            log_template.append(command_keyword)
        else:
            for leaf in last_tree_position:
                if parameter_regex.match(leaf) is not None:
                    if leaf in last_tree_position:
                        last_tree_position = last_tree_position[leaf]  # Save the last state of the tree to travel on it
                        # log_template.append('<*>')
                        log_template.append(leaf)

    return ' '.join(log_template)


# computes the k-means clustering and the score of the algorithm
def compute_kmeans_score(X, kmin, kmax, document, folder, step, plot_clustering_score, eval_method):
    sil = []
    var = []
    models = []
    # the k-means is done using the values of k from 'kmin' to 'kmax' with step 'step'
    for k in range(kmin, kmax + 1, step):
        # if k is greater than the number of samples the k-means can be done
        if k < X.shape[0]:

            # k-means execution
            model = compute_kmeans(X, k, document)
            labels = model.labels_  # label for each sample

            # the silhouette score can be avoided
            if compute_silhouette_score:  # silhouette score is too heavy
                print('Computing silhouette score using k =', k, 'for:', document)
                # the silhouette score is computed and stored into an array
                sil_score = silhouette_score(X, labels, metric='euclidean')
                sil.append(sil_score)
            else:
                # if the silhouette score isn't computed the array is filled with None
                sil.append(None)
            # the variance is stored into an array
            var.append(model.inertia_)
            # the model is stored into an array
            models.append(model)
        else:
            # if k is greater than the number of samples 'kmax' is set to k - 1
            print('Max number of clusters.')
            kmax = k - 1
            break

    # the knee of the variance plot is found
    kn = KneeLocator(range(kmin, kmax + 1, step), var, curve='convex', direction='decreasing', interp_method='interp1d')

    # computation of the clustering score
    if plot_clustering_score:
        compute_kmeans_score_plot(kmin, kmax, sil, var, document, folder, step, kn)

    best_model = None
    # if the evaluation method is the silhouette score the best k is chosen by using the greatest
    # value of the silhouette score in the 'sil' array
    if eval_method == 'Silhouette Score':
        max_score = max(sil)
        max_score_index = sil.index(max_score)
        # the best model is selected using the index of the best value of the silhouette score
        best_model = models[max_score_index]  # using silhouette score
        k_best = range(kmin, kmax + 1, step)[max_score_index]
    elif kn.knee is None:  # if the variance plot has no elbow than the best number of clusters is the lowest
        best_model = models[0]
        k_best = range(kmin, kmax + 1, step)[0]
    elif eval_method == 'Elbow Method':
        # the best model is selected using the index of the knee - 2 because the clusters start from 2
        best_model = models[kn.knee - 2]
        k_best = kn.knee  # the best k is the knee

    # if the knn plot has no elbow, the biggest eps is used
    if best_model is None:
        best_model = models[-1]

    return [k_best, best_model, models, best_model.labels_, sil, var, kn]


# makes the plot of the silhouete score, the variance and the position of the elbow
def compute_kmeans_score_plot(kmin, kmax, sil, var, document, folder, step, kn):
    print('Computing k score plot for {}'.format(document))

    plt.figure()

    fig, ax = plt.subplots()
    p1 = ax.plot(range(kmin, kmax + 1, step), sil, color='red')
    plt.title('Silhouette scores for\n {}'.format(document))
    plt.xlabel('Clusters')
    ax.set_ylabel('Silhouette score')

    ax2 = ax.twinx()
    p2 = ax2.plot(range(kmin, kmax + 1, step), var, color='blue')
    ax2.set_ylabel('Variance')
    plt.xticks(range(2, kmax + 1, step))

    # if the knee is present a dashed line that represent the value of the knee is added to the plot
    if kn.knee is not None:
        plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    else:
        print('{} variance plot has no Elbow'.format(document))

    line_labels = ["Silhouette score", "Variance"]
    fig.legend([p1, p2],  # The line objects
               labels=line_labels,  # The labels for each line
               loc="lower right",  # Position of legend
               ncol=2,
               # bbox_to_anchor=[0, ax.get_position().y0 - 0.06, 1, 1], bbox_transform=fig.transFigure,
               borderaxespad=0.1,  # Small spacing around legend box
               # title="Legend Title"  # Title for the legend
               )

    # plt.show()
    plt.savefig("plots/{}/clustering scores/{}.{}".format(folder, document, FORMAT), format=FORMAT)
    plt.close()

    return plt


# similar to compute_kmeans_score
def compute_dbscan_score(X, epsmin, epsmax, document, folder, step, plot_clustering_score, eval_method):
    sil = []
    var = []
    models = []
    # the DBSCAN is done using different values of eps, the distance between the points in a dense region
    for eps in np.arange(epsmin, epsmax, step):

        # dbscan execution
        model = compute_dbscan(X, eps, document)
        labels = model.labels_  # label for each sample

        # the silhouette score can be avoided
        if compute_silhouette_score:
            print('Computing silhouette score using eps =', eps, 'for:', document)
            sil_score = silhouette_score(X, labels, metric='euclidean')
            sil.append(sil_score)
        else:
            # if the silhouette score isn't computed the array is filled with None
            sil.append(None)
        var.append(None)
        models.append(model)

    kn = knn_score(X)  # the knee is computed by using the k-nearest neighbors

    # computation of the clustering score
    if plot_clustering_score:
        compute_dbscan_score_plot(epsmin, epsmax, sil, document, folder, step, kn)

    # if the evaluation method is the silhouette score the best k is chosen by using the geatest
    # value of the silhouette score in the 'sil' array
    if eval_method == 'Silhouette Score':
        max_score = max(sil)
        max_score_index = sil.index(max_score)
        # the best model is selected using the index of the best value of the silhouette score
        best_model = models[max_score_index]  # using silhouette score
        eps_best = np.arange(epsmin, epsmax + 1, step)[max_score_index]
    elif kn.knee is None:  # if the variance plot has no elbow than the best number of clusters is the lowest
        best_model = models[0]
        eps_best = range(epsmin, epsmax, step)[0]
    elif eval_method == 'Elbow Method':
        eps_best = round(kn.knee_y, round_precision)
        for model in models:
            # for every model the best eps and the eps of the model are compared to choose the
            # model with the best eps
            model_eps = round(model.eps, round_precision)
            if eps_best == model_eps:
                best_model = model
                break

    return [eps_best, best_model, models, best_model.labels_, sil, var, kn]


# similar to compute_kmeans_plot
def compute_dbscan_score_plot(epsmin, epsmax, sil, document, folder, step, kn):
    print('Computing eps score plot for\n {}'.format(document))

    plt.figure()

    fig, ax = plt.subplots()
    ax.plot(np.arange(epsmin, epsmax, step), sil, color='red')
    plt.title('Silhouette scores for \n {}'.format(document))
    plt.xlabel('eps')
    ax.set_ylabel('Silhouette score')

    # if the knee is present a dashed line that represent the value of the knee is added to the plot
    if kn.knee is not None:
        plt.vlines(round(kn.knee_y, round_precision), plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    else:
        print('{} has no Elbow'.format(document))

    # plt.show()
    plt.savefig("plots/{}/clustering scores/{}.{}".format(folder, document, FORMAT), format=FORMAT)
    plt.close()

    return plt


# finds the elbow in the knn plot
def knn_score(X):

    # the knn is used to make a plot and find the knee
    neighbors = 6
    nbrs = NearestNeighbors(n_neighbors=neighbors).fit(X)
    distances, indices = nbrs.kneighbors(X)
    distance_desc = sorted(distances[:, neighbors - 1], reverse=True)
    # plt.plot(list(range(1, len(distance_desc) + 1)), distance_desc)
    # plt.show()
    kn = KneeLocator(range(1, len(distance_desc) + 1),  # x values
                     distance_desc,  # y values
                     S=1.0,  # parameter suggested from paper
                     curve="convex",  # parameter from figure
                     direction="decreasing")  # parameter from figure
    # plt.close()

    return kn


def compute_kmeans(X, k, document):
    print('Computing k-means score using k =', k, 'for:', document)
    # define the model
    model = KMeans(n_clusters=k)
    # fit the model
    model.fit(X)
    # assign a cluster to each example
    # yhat = model.predict(X)

    return model


def compute_dbscan(X, eps, document):
    print('Computing DBSCAN for:', document)
    model = DBSCAN(eps)
    model.fit(X)
    # labels = model.labels_ to return labels

    return model


def compute_NMF_components(X):
    n_components = 20
    nmf = NMF(n_components)
    coords = nmf.fit_transform(np.asarray(X))

    return coords


def compute_SVD_components(X):
    X = csr_matrix(X)
    svd = TruncatedSVD(n_components=20, n_iter=7, random_state=42)
    coords = svd.fit_transform(X)

    return coords


# the correlation matrix represents the correlation between the original variables and the
# new transformed variables
def heatmap_pca(pca, features, document, folder):
    print('Computing heatmap for {}'.format(document))
    fig = plt.figure()
    pca_comp = pca.components_
    pca_comp = pca_comp.T
    plt.title('Correlation between original and\n transformed features for\n {}'.format(document))
    plt.imshow(pca_comp, cmap="RdYlBu")
    plt.colorbar()
    plt.yticks(range(len(features)), features)
    plt.xticks([*range(0, pca.n_components_)])
    axis = plt.gca()
    axis.set_aspect('auto')
    max_yticks = 30
    yloc = plt.MaxNLocator(max_yticks)
    axis.yaxis.set_major_locator(yloc)
    plt.tight_layout()
    # plt.show()
    plt.savefig("plots/{}/heatmaps/{}.{}".format(folder, document, FORMAT), format=FORMAT)
    plt.close()

    return plt


# histogram of the percentage of variance explained by every component
def variance_explained(pca, document, folder):
    print('Computing variance for {}'.format(document))
    plt.figure()
    PC = range(0, pca.n_components_)
    plt.bar(PC, pca.explained_variance_ratio_)
    plt.plot(np.cumsum(pca.explained_variance_ratio_), color='red')
    plt.title('Variance explained ratio for\n {}'.format(document))
    plt.xlabel('Principal Components')
    plt.ylabel('Variance %')
    plt.xticks(PC)
    plt.legend(["Cumulative sum of the variace", "Percentage of variance explained"])
    plt.gca().yaxis.grid(True)
    # plt.show()
    plt.savefig("plots/{}/variance explained ratio/{}.{}".format(folder, document, FORMAT), format=FORMAT)
    plt.close()

    return plt


# scatter matrix to represent the components
def compute_scatter_matrix(coords_df, document, folder):
    print('Computing scatter matrix for {}'.format(document))
    # fig = px.scatter_matrix(coords_df, dimensions=[*range(0, pca.n_components_)], color="Cluster", title=document)
    # fig = pd.plotting.scatter_matrix(coords_df, diagonal='kde')
    # fig = sns.pairplot(coords_df, hue="Cluster", palette="bright")
    fig = sns.PairGrid(coords_df, hue="Cluster", palette="bright", corner=True)
    fig = fig.map(sns.scatterplot)
    # fig.map_diag(sns.histplot)
    # fig.map_offdiag(sns.scatterplot)
    fig.savefig("plots/{}/scatter matrices/{}.{}".format(folder, document, FORMAT), format=FORMAT)

    return fig


# bar chart of a single cluster for a profilename, testcase or both
def plot_bar_chart(keys, values, field, log_file, name, color, n):

    # the commands are sorted alphabetically
    df_bar_chart = pd.DataFrame({'keys': keys, 'lengths': values})
    df_bar_chart = df_bar_chart.sort_values(by='keys', ascending=False)

    plt.figure()
    # same clusters are represented with same colors in the total bar chart
    plt.barh(df_bar_chart['keys'], df_bar_chart['lengths'], color=color)
    plt.yticks(range(len(df_bar_chart['keys'])), df_bar_chart['keys'], fontsize=9)
    plt.title('Commands number for \n cluster {}'.format(n))
    axis = plt.gca()
    axis.set_aspect('auto')
    if len(df_bar_chart['keys']) > 30:
        max_yticks = 30
        yloc = plt.MaxNLocator(max_yticks)
        axis.yaxis.set_major_locator(yloc)
    plt.tight_layout()
    plt.gca().xaxis.grid(True)
    plt.savefig("plots/{}/visualization/{}.{}".format(field + '_' + log_file, name, FORMAT),
                format=FORMAT)
    plt.close()
    # plt.show()


# plots the commands in a cluster, in load_workspace is used to represent the number of commands or the
# number of different commands
def plot_commands_per_cluster(cmd_per_cluster, cluster_names, cluster_colors, field, log_file, name):
    df_bar_chart = pd.DataFrame({'keys': cmd_per_cluster, 'clusters': cluster_names, 'colors': cluster_colors})

    plt.figure()
    # the bar chart uses different colors for each cluster
    plt.bar(df_bar_chart['clusters'], df_bar_chart['keys'], color=df_bar_chart['colors'])
    plt.xticks(range(len(df_bar_chart['clusters'])), df_bar_chart['clusters'], fontsize=9)
    plt.title('Commands number for\n {}'.format(name))
    axis = plt.gca()
    axis.set_aspect('auto')
    if len(df_bar_chart['keys']) > 30:
        max_yticks = 30
        yloc = plt.MaxNLocator(max_yticks)
        axis.yaxis.set_major_locator(yloc)
    plt.tight_layout()
    plt.gca().yaxis.grid(True)
    plt.savefig("plots/{}/visualization/{}.{}".format(field + '_' + log_file, name, FORMAT),
                format=FORMAT)
    plt.close()
    # plt.show()

# bar chart of the entire profilename, testcase or both
def plot_bar_chart_total(keys, values, cluster_colors, legend_list, field, log_file, name):
    df_bar_chart = pd.DataFrame({'keys': keys, 'lengths': values, 'colors': cluster_colors})
    # the commands are sorted alphabetically
    df_bar_chart = df_bar_chart.sort_values(by='keys', ascending=False)

    plt.figure()
    # the bar chart uses different colors for each cluster
    plt.barh(df_bar_chart['keys'], df_bar_chart['lengths'], color=df_bar_chart['colors'])
    plt.yticks(range(len(df_bar_chart['keys'])), df_bar_chart['keys'], fontsize=9)
    plt.title('Commands number for\n {}'.format(name))
    plt.legend(handles=legend_list, loc="upper right", shadow=True, title='Clusters')
    axis = plt.gca()
    axis.set_aspect('auto')
    if len(df_bar_chart['keys']) > 30:
        max_yticks = 30
        yloc = plt.MaxNLocator(max_yticks)
        axis.yaxis.set_major_locator(yloc)
    plt.tight_layout()
    plt.gca().xaxis.grid(True)
    plt.savefig("plots/{}/visualization/{}.{}".format(field + '_' + log_file, name, FORMAT),
                format=FORMAT)
    plt.close()
    # plt.show()


# bar chart of a single cluster for a profilename, testcase or both
def plot_tfidf_chart(keys, values, field, log_file, name, color, maximum):

    if maximum >= len(keys):
        maximum = len(keys) - 1

    df_bar_chart = pd.DataFrame({'keys': keys, 'lengths': values})
    df_bar_chart = df_bar_chart.sort_values(by='lengths', ascending=False)

    plt.figure()
    # same clusters are represented with same colors in the total bar chart, only the first 'maximum' are shown
    plt.barh(df_bar_chart['keys'][0:maximum+1], df_bar_chart['lengths'][0:maximum+1], color=color)
    plt.title('TF-IDF values for\n cluster {}'.format(name))
    axis = plt.gca()
    axis.set_aspect('auto')
    if len(df_bar_chart['keys'][0:maximum+1]) > 30:
        max_yticks = 30
        yloc = plt.MaxNLocator(max_yticks)
        axis.yaxis.set_major_locator(yloc)
    plt.tight_layout()
    plt.gca().xaxis.grid(True)
    plt.savefig("plots/{}/visualization/{}.{}".format(field + '_' + log_file, name, FORMAT),
                format=FORMAT)
    plt.close()
    # plt.show()


# bar chart of the entire profilename, testcase or both
def plot_tfidf_chart_total(keys, values, cluster_colors, legend_list, field, log_file, name, maximum):

    if maximum >= len(keys):
        maximum = len(keys) - 1

    df_bar_chart = pd.DataFrame({'keys': keys, 'lengths': values, 'colors': cluster_colors})
    df_bar_chart = df_bar_chart.sort_values(by='lengths', ascending=False)

    plt.figure()
    # the bar chart uses different colors for each cluster, only the first 'maximum' are shown
    plt.barh(df_bar_chart['keys'][0:maximum+1], df_bar_chart['lengths'][0:maximum+1], color=df_bar_chart['colors'])
    plt.title('TF-IDF values for\n {}'.format(name))
    plt.legend(handles=legend_list, loc="upper right", shadow=True, title='Clusters')
    axis = plt.gca()
    axis.set_aspect('auto')
    if len(df_bar_chart['keys'][0:maximum+1]) > 30:
        max_yticks = 30
        yloc = plt.MaxNLocator(max_yticks)
        axis.yaxis.set_major_locator(yloc)
    plt.tight_layout()
    plt.gca().xaxis.grid(True)
    plt.savefig("plots/{}/visualization/{}.{}".format(field + '_' + log_file, name, FORMAT),
                format=FORMAT)
    plt.close()
    # plt.show()


# correlation matrix for a profilename, testcase or both
def compute_correlation_matrix(df_tfidfvect, document, folder):
    print('Computing correlation matrix for {}'.format(document))
    plt.figure()
    plt.title('Correlation matrix for\n {}'.format(document))
    plt.imshow(df_tfidfvect.corr(), cmap="RdYlBu")
    plt.colorbar()
    # plt.xticks(range(len(df_tfidfvect.shape[1])), df_tfidfvect.columns, rotation=90)
    # plt.yticks([*range(0, pca.n_components_)])
    # plt.show()
    plt.savefig("plots/{}/correlation matrices/{}.{}".format(folder, document, FORMAT), format=FORMAT)
    plt.close()

    return plt


# covariance matrix for a profilename, testcase or both
def compute_covariance_matrix(df_tfidfvect, document, folder):
    print('Computing covariance matrix for {}'.format(document))
    plt.figure()
    plt.title('Covariance matrix for\n {}'.format(document))
    plt.imshow(df_tfidfvect.cov(), cmap="RdYlBu")
    plt.colorbar()
    # plt.xticks(range(len(df_tfidfvect.shape[1])), df_tfidfvect.columns, rotation=90)
    # plt.yticks([*range(0, pca.n_components_)])
    # plt.show()
    plt.savefig("plots/{}/covariance matrices/{}.{}".format(folder, document, FORMAT), format=FORMAT)
    plt.close()

    return plt


def save_parameters(parameters, filename):
    print('Saving parameters...')
    with open(filename, 'wb') as f:
        pickle.dump(parameters, f)


def load_parameters(field, log_file, filename):
    with open(filename, 'rb') as f:
        [MIN_VALUE,
         MAX_VALUE,
         MIN_NGRAMS,
         MAX_NGRAMS,
         EVAL_METHOD,
         CLUSTERING_ALG,
         STEP,
         DIM_RED_METHOD] = pickle.load(f)

    parameters = {'Field': field,
                  'Log name': log_file,
                  'Min param value': MIN_VALUE,
                  'Max param value': MAX_VALUE,
                  'Min N-grams': MIN_NGRAMS,
                  'Max N-grams': MAX_NGRAMS,
                  'Evaluation method': EVAL_METHOD,
                  'Step used for parameter': STEP,
                  'Clustering algorithm': CLUSTERING_ALG,
                  'Dimensionality reduction method': DIM_RED_METHOD}

    return parameters


def save_clustering_results(variables, field, file, document, results_path):
    print('Saving {}...'.format(document))
    filename = '{}/{}_variables_{}_{}.pkl'.format(results_path, field, file, document)
    with open(filename, 'wb') as f:
        pickle.dump(variables, f)


def load_clustering_results(results, field, log_file):
    doc_per_profile = []  # contains all the commands for every profile
    features_per_doc = []  # contains all the features for every profile
    cluster_labels = []  # contains the labels for every entry in each profile
    models = []  # information about the ML model used for every profile
    models_per_profile = []  # list of every ML model used for each profile
    Xs = []  # input transformed in TF-IDF
    silhouette_scores = []  # silhouette scores for MIN_CLUSTERS <= k <= MAX_CLUSTERS for every profile
    variance_scores = []  # variance for MIN_CLUSTERS <= k <= MAX_CLUSTERS for every profile
    knees = []  # contains all the knees values obtained from the elbow method
    dim_red_array = []  # contains the PCAs of every profile
    coords_vect = []  # results of the PCA for every profile
    names = []  # profile names

    file_list = os.listdir('workspace/{}'.format(results))
    target = '{}_variables_{}'.format(field, log_file)

    # variables are loaded and organized in a DataFrame to be used in load_workspace
    for file in file_list:
        if file.startswith(target):
            filename = 'workspace/{}/{}'.format(results, file)
            print('loading {}'.format(file))
            with open(filename, 'rb') as f:
                [name,
                 document_series,
                 X,
                 features,
                 dim_red,
                 coords,
                 labels,
                 model,
                 models_list,
                 sil,
                 var,
                 kn] = pickle.load(f)

                names.append(name)
                doc_per_profile.append(document_series)
                Xs.append(X)
                features_per_doc.append(features)
                dim_red_array.append(dim_red)
                coords_vect.append(coords)
                cluster_labels.append(labels)
                models.append(model)
                models_per_profile.append(models_list)
                silhouette_scores.append(sil)
                variance_scores.append(var)
                knees.append(kn)

    df = pd.DataFrame({'Names': names,
                       'Input data': doc_per_profile,
                       'TF-IDF data': Xs,
                       'Features': features_per_doc,
                       'Dimensionality reduction': dim_red_array,
                       'Transformed components': coords_vect,
                       'Labels': cluster_labels,
                       'Best ML model': models,
                       'Other models': models_per_profile,
                       'Silhouette scores': silhouette_scores,
                       'Variance': variance_scores,
                       'Knee': knees})
    df = df.set_index('Names')

    return df


# clutering
def compute_clustering(documents, document, vect, field, file, CLUSTERING_ALG, MIN_VALUE, MAX_VALUE, STEP,
                       EVAL_METHOD, DIM_RED_METHOD, plot_clustering_score, plot_heatmap, plot_variance,
                       plot_scatter_matrix, save_variables, results_path, document_series):

    # if a profilename, testcase or both has length equal to 0 is skipped
    if len(documents[document]) > 0:

        # learn vocabulary and idf, return document-term matrix
        X = vect.fit_transform(document_series)
        # print(X)
        # if there are 3 or less samples the profilename, testcase or both is skipped
        if X.shape[0] > 3:
            print('{} {} has {} lines'.format(field, document, len(document_series)))

            # get output feature names for transformation
            features = vect.get_feature_names_out()
            print('{} {} has {} features'.format(field, document, len(features)))

            # dataframe with features and TF-IDF values
            # df_tfidfvect = pd.DataFrame(data=X.toarray(), columns=features)
            # print(df_tfidfvect)

            X_dense = X.todense()
            # print(X_dense)

            # the dimensionality reduction method is selected
            if DIM_RED_METHOD == 'PCA':
                dim_red = PCA(n_components=0.95)
                coords = dim_red.fit_transform(np.asarray(X_dense))
            elif DIM_RED_METHOD == 'SparsePCA':
                dim_red = SparsePCA(n_components=20)
                coords = dim_red.fit_transform(np.asarray(X_dense))
            elif DIM_RED_METHOD == 'NMF':
                dim_red = NMF(n_components=20)
                coords = dim_red.fit_transform(np.asarray(X_dense))
            print('{} {} has {} components'.format(field, document, dim_red.n_components_))

            # the clustering algorithm is selected
            if CLUSTERING_ALG == 'K-Means':
                [k, model, models_list, yhat, sil, var, kn] = compute_kmeans_score(
                    coords, MIN_VALUE, MAX_VALUE, document, field + '_' + file, STEP, plot_clustering_score,
                    EVAL_METHOD)
                print('Number of clusters:', k, 'using', EVAL_METHOD)
            elif CLUSTERING_ALG == 'DBSCAN':
                [eps, model, models_list, yhat, sil, var, kn] = compute_dbscan_score(
                    coords, MIN_VALUE, MAX_VALUE, document, field + '_' + file, STEP, plot_clustering_score,
                    EVAL_METHOD)
                print('eps used:', eps, 'created', len(unique(yhat)), 'clusters using', EVAL_METHOD)

            # dataframe containing the commands and their labels in the cluster od
            # cmd_df = pd.DataFrame({'cmd': document_series, 'Cluster': yhat})

            # dataframe containing the transformed variables and their cluster id
            coords_df = pd.DataFrame(coords)
            coords_df['Cluster'] = yhat

            if plot_heatmap:
                heatmap_pca(dim_red, features, document, field + '_' + file)
            if plot_variance:
                variance_explained(dim_red, document, field + '_' + file)
            if plot_scatter_matrix:
                compute_scatter_matrix(coords_df, document, field + '_' + file)

            # the results are saved so they can be loaded to make other processing
            if save_variables:
                variables = [document,
                             document_series,
                             X,
                             features,
                             dim_red,
                             coords,
                             yhat,
                             model,
                             models_list,
                             sil,
                             var,
                             kn]

                save_clustering_results(variables, field, file, document, results_path)
        else:
            # if a document is too small the clustering computation is skipped
            print('{} is too small.'.format(document))
