import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import unique
import matplotlib.patches as mpatches

import functions

field = 'profilename'  # profilename, testcase, profilename_testcase
log_file = '2021-05-01'  # log file to analyse
format = 'png'
results = 'results 2021-05-01 (k, 1-5 ngrams, pca)'  # folder containing the results

n_results = 3  # number of profilenames, testcases or a combination of both to analyze
n_commands = 10  # number of commands to show in the TF-IDF bar chart for a single cluster
n_commands_2 = 20  # number of commands to show in the TF-IDF bar chart for a profile, testcase or both
show_results = True  # if the results of clustering will be computed

# if the plots will be computed
plot_kmeans_score = True
plot_dbscan_score = False
plot_heatmap = True
plot_variance = True
plot_scatter_matrix = True
plot_correlation_matrix = True
plot_correlation_matrix_reduced = True
plot_covariance_matrix = True
plot_covariance_matrix_reduced = True

# the folder containing the plots are created if necessary
path = 'plots/{}_{}'.format(field, log_file)
if not os.path.exists(path):
    os.makedirs(path)
if plot_heatmap and not os.path.exists('{}/heatmaps'.format(path)):
    os.makedirs('{}/heatmaps'.format(path))
if plot_scatter_matrix and not os.path.exists('{}/scatter matrices'.format(path)):
    os.makedirs('{}/scatter matrices'.format(path))
if (plot_kmeans_score or plot_dbscan_score) and not os.path.exists('{}/clustering scores'.format(path)):
    os.makedirs('{}/clustering scores'.format(path))
if plot_variance and not os.path.exists('{}/variance explained ratio'.format(path)):
    os.makedirs('{}/variance explained ratio'.format(path))
if plot_correlation_matrix and not os.path.exists('{}/correlation matrices'.format(path)):
    os.makedirs('{}/correlation matrices'.format(path))
if plot_covariance_matrix and not os.path.exists('{}/covariance matrices'.format(path)):
    os.makedirs('{}/covariance matrices'.format(path))
if show_results and not os.path.exists('{}/visualization'.format(path)):
    os.makedirs('{}/visualization'.format(path))
print('plots folder created')

# the parameters are loaded
filename = 'workspace/{}/parameters.pkl'.format(results)
parameters = functions.load_parameters(field, log_file, filename)

# the results are loaded and sorted according to the number of samples
df = functions.load_clustering_results(results, field, log_file)
df['Length'] = 0
n = 0
while n < len(df):
    df['Length'][n] = len(df['Input data'][n])
    n += 1
df = df.sort_values(by='Length', ascending=False)

save_path = 'plots/{}_{}/visualization'.format(field, log_file)


# plots the commands' count for every cluster and makes a bar chart showing their distribution
def plot_command_count_results(df, n_results):
    i = 0

    while i < n_results:
        # a target dataframe is created containing the input data and the labels
        target = pd.DataFrame({'cmd': df.iloc[i]['Input data'], 'Label': df.iloc[i]['Labels']})
        n_clusters = max(target['Label'])
        colors = [plt.cm.tab20(np.arange(20))]
        colors2 = [plt.cm.tab20b(np.arange(20))]
        colors = [color for color in colors[0]]
        colors2 = [color for color in colors2[0]]
        colors += colors2

        keys = []
        lengths = []
        clusters = []
        cluster_colors = []
        cluster_names = []
        cmd_per_cluster = []
        cmd_names_per_cluster = []
        n = 0
        while n <= n_clusters:
            # for every cluster the commands are grouped and counted
            df_group = target[target['Label'] == n]
            df_cmd = df_group.groupby(df_group['cmd'])
            cluster_keys = list(df_cmd.groups.keys())
            cluster_values = [len(value) for value in df_cmd.groups.values()]
            cmd_per_cluster.append(sum(cluster_values))
            cmd_names_per_cluster.append(len(df_cmd))

            functions.plot_bar_chart(cluster_keys, cluster_values, field, log_file,
                                     '{}_cluster_{}'.format(df.iloc[i].name, n), colors[n], n)

            # the commands, their number and their clusters are stored in some arrays
            keys += list(df_cmd.groups.keys())
            lengths += [len(value) for value in df_cmd.groups.values()]
            clusters += [n] * len(list(df_cmd.groups.keys()))
            cluster_colors += [colors[n]] * len(list(df_cmd.groups.keys()))
            cluster_names.append(n)

            n += 1

        legend_list = []
        for cluster, color in zip(cluster_names, colors):
            legend_list.append(mpatches.Patch(color=color, label=cluster))

        functions.plot_commands_per_cluster(cmd_per_cluster, cluster_names, colors[0:n_clusters+1], field, log_file,
                                            df.iloc[i].name + '_cmd_number')
        functions.plot_commands_per_cluster(cmd_names_per_cluster, cluster_names, colors[0:n_clusters+1], field, log_file,
                                            df.iloc[i].name + '_cmd_count')
        functions.plot_bar_chart_total(keys, lengths, cluster_colors, legend_list, field, log_file,
                                       df.iloc[i].name)

        i += 1


# plots the highest TF-IDF values
def plot_tfidf_results(df, n_results, n_commands, n_commands_2):
    i = 0
    for profile in df.index:
        X = df['TF-IDF data'].loc[profile]
        features = df['Features'].loc[profile]
        labels = df['Labels'].loc[profile]
        n_clusters = max(unique(labels))
        colors = [plt.cm.tab20(np.arange(n_clusters + 1))]
        colors = [color for color in colors[0]]

        c = 0
        total_cmd_keys = []
        total_cmd_tfidf = []
        cluster_colors = []
        cluster_names = []
        while c <= n_clusters:
            keywords = []
            tf_idfs = []
            n = 0
            # for every value of TF-IDF different from 0 the corresponding feature is found
            # and stored into an array along with the TF-IDF value
            while n < X.shape[0]:
                if labels[n] == c:
                    feature_index = X[n, :].nonzero()[1]
                    tfidf_scores = zip(feature_index, [X[n, x] for x in feature_index])
                    for w, s in [(features[i], s) for (i, s) in tfidf_scores]:
                        keywords.append(w)
                        tf_idfs.append(s)
                n += 1

            cluster_names.append(c)

            # a DataFrame containing the keywords and the tf-idf values of a single cluster is created
            # and grouped by keyword and tf-idf value
            tfidf_keywords_value = pd.DataFrame({'keyword': keywords, 'tf-idf': tf_idfs})
            tfidf_keywords_groups = tfidf_keywords_value.groupby(by=['keyword', 'tf-idf'])
            # the DataFrame is converted into tuples and an array ov colors is created
            tfidf_keywords_groups = [key for key, value in tfidf_keywords_groups.groups.items()]
            tfidf_keywords_groups.sort(key=lambda y: y[1], reverse=True)
            cmd_keys = [t[0] for t in tfidf_keywords_groups]
            cmd_tfidf = [t[1] for t in tfidf_keywords_groups]
            # the arrays containing the commands and the tf-idf values of all the cluster are used to
            # plot a bar chart
            total_cmd_keys += cmd_keys
            total_cmd_tfidf += cmd_tfidf
            cluster_colors += [colors[c]] * len(cmd_keys)

            # the array are passed to the function to plot a bar chart of the 'n_commands' with the
            # biggest tf-idf value o the cluster
            functions.plot_tfidf_chart(cmd_keys, cmd_tfidf, field, log_file,
                                       profile + '-TF-IDF_cluster_{}'.format(c), colors[c], n_commands)
            c += 1

        legend_list = []
        for cluster, color in zip(cluster_names, colors):
            legend_list.append(mpatches.Patch(color=color, label=cluster))

        # the bar chart of all the clusters of a single profilename, testcase or both is made using all the
        # commands and tf-idf values
        functions.plot_tfidf_chart_total(total_cmd_keys, total_cmd_tfidf, cluster_colors, legend_list,
                                         field, log_file, profile + '-TF-IDF', n_commands_2)

        i += 1
        if i == n_results:
            break


# plots the number of commands present in every profile or testcase in a field
def plot_field(df):
    plt.figure()
    plt.barh(df.index, df['Length'])
    plt.title('Number of commands for {}'.format(field))
    axis = plt.gca()
    # axis.set_aspect('auto')
    if len(df.index) > 30:
        max_yticks = 30
        yloc = plt.MaxNLocator(max_yticks)
        axis.yaxis.set_major_locator(yloc)
    plt.tight_layout()
    plt.savefig("plots/{}/visualization/{}.{}".format(field + '_' + log_file, log_file, format),
                format=format)
    plt.close()


# Plot k-means score
if plot_kmeans_score:
    i = 0
    # the k-means score plot is computed for the biggest 'n_results' profilename, testcase or both
    for profile in df.index:
        sil = df['Silhouette scores'].loc[profile]
        var = df['Variance'].loc[profile]
        kn = df['Knee'].loc[profile]
        score_plot = functions.compute_kmeans_score_plot(
            parameters['Min param value'],
            parameters['Max param value'],
            sil,
            var,
            profile,
            parameters['Field'] + '_' + parameters['Log name'],
            parameters['Step used for parameter'],
            kn)

        i += 1
        if i == n_results:
            break

# Plot DBSCAN score
if plot_dbscan_score:
    # the dbscan score plot is computed for the biggest 'n_results' profilename, testcase or both
    i = 0
    for profile in df.index:
        sil = df['Silhouette scores'].loc[profile]
        X = df['Input data'].loc[profile]
        kn = functions.knn_score(X)
        score_plot = functions.compute_dbscan_score_plot(
            parameters['Min param value'],
            parameters['Max param value'],
            sil,
            profile,
            parameters['Field'] + '_' + parameters['Log name'],
            parameters['Step used for parameter'],
            kn)

        i += 1
        if i == n_results:
            break

# Plot heatmaps
if plot_heatmap:
    # the correlation between original and transformed variables plot is computed for the biggest
    # 'n_results' profilename, testcase or both
    i = 0
    for profile in df.index:
        pca = df['Dimensionality reduction'].loc[profile]
        features = df['Features'].loc[profile]
        heatmap_plot = functions.heatmap_pca(pca, features, profile, parameters['Field'] + '_' + parameters['Log name'])

        i += 1
        if i == n_results:
            break

# Plot variance explained
if plot_variance:
    # the variance explained ratio plot is computed for the biggest 'n_results' profilename, testcase or both
    i = 0
    for profile in df.index:
        pca = df['Dimensionality reduction'].loc[profile]
        variance_histogram = functions.variance_explained(pca, profile,
                                                          parameters['Field'] + '_' + parameters['Log name'])

        i += 1
        if i == n_results:
            break

# Plot scatter matrix
if plot_scatter_matrix:
    # the the scatter matrix is computed for the biggest 'n_results' profilename, testcase or both
    i = 0
    for profile in df.index:
        coords_df = pd.DataFrame(df['Transformed components'].loc[profile])
        clusters = df['Labels'].loc[profile]
        coords_df['Cluster'] = clusters
        fig = functions.compute_scatter_matrix(coords_df, profile, parameters['Field'] + '_' + parameters['Log name'])

        i += 1
        if i == n_results:
            break

if plot_correlation_matrix:
    # the correlation matrix between the original features is computed for the biggest 'n_results'
    # profilename, testcase or both
    i = 0
    for profile in df.index:
        X = df['TF-IDF data'].loc[profile]
        features = df['Features'].loc[profile]
        df_tfidfvect = pd.DataFrame(data=X.toarray(), columns=features)
        functions.compute_correlation_matrix(df_tfidfvect, profile, parameters['Field'] + '_' + parameters['Log name'])

        i += 1
        if i == n_results:
            break

if plot_correlation_matrix_reduced:
    # the correlation matrix between the principal components is computed for the biggest 'n_results'
    # profilename, testcase or both
    i = 0
    for profile in df.index:
        coords = df['Transformed components'].loc[profile]
        df_coords = pd.DataFrame(data=coords)
        functions.compute_correlation_matrix(df_coords, profile + '_reduced',
                                             parameters['Field'] + '_' + parameters['Log name'])

        i += 1
        if i == n_results:
            break

if plot_covariance_matrix:
    # the covariance matrix between the original features is computed for the biggest 'n_results'
    # profilename, testcase or both
    i = 0
    for profile in df.index:
        X = df['TF-IDF data'].loc[profile]
        features = df['Features'].loc[profile]
        df_tfidfvect = pd.DataFrame(data=X.toarray(), columns=features)
        functions.compute_covariance_matrix(df_tfidfvect, profile, parameters['Field'] + '_' + parameters['Log name'])

        i += 1
        if i == n_results:
            break

if plot_covariance_matrix_reduced:
    # the correlation matrix between the principal components is computed for the biggest 'n_results'
    # profilename, testcase or both
    i = 0
    for profile in df.index:
        coords = df['Transformed components'].loc[profile]
        df_coords = pd.DataFrame(data=coords)
        functions.compute_covariance_matrix(df_coords, profile + '_reduced',
                                            parameters['Field'] + '_' + parameters['Log name'])

        i += 1
        if i == n_results:
            break

# the bar chart are made and saved
if show_results:
    print('Computing results bar charts...')
    plot_command_count_results(df, n_results)
    plot_tfidf_results(df, n_results, n_commands, n_commands_2)
    plot_field(df)

print('end')
