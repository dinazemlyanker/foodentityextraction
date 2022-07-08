import math
import pandas as pd
import pickle
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from clustering import findCluster
from clustering import find_optimal_clusters
from clustering import convert_to_1d
from clustering import create_graph_dicts
from clustering import create_distance_matrix
from clustering import get_df_shared_ingredients_network
from clustering import visualize_networks
from clustering import get_subnets
import networkx as nx

def convert_cluster_list_to_dfs(initial_df, cluster_list, id_list, timing):
    for cluster_index in range(len(cluster_list)):
        recipe_id_list = []
        cluster = cluster_list[cluster_index]
        for recipe_index in cluster:
            recipe_id = id_list[recipe_index]
            recipe_id_list.append(recipe_id)
        cluster_df = initial_df[initial_df['story_id'] in recipe_id_list]
        name_of_df = timing + str(cluster_index) + '_cluster_df.csv'
        cluster_df.to_csv(name_of_df)

if False:
    # divide this into before and after the date
    article_df = pd.read_csv('cleaned_stratified.csv')
    article_df_before = article_df[article_df['date'] < '2020-06-01']
    article_df_before_samp = article_df_before.sample(n=200)
    article_df_after = article_df[article_df['date'] > '2020-06-01']
    article_df_after_samp = article_df_after.sample(n=200)



    # create graphs in the form of dictionaries
    id_to_food_dict_bef, food_to_id_dict_bef, id_to_food_count_bef = create_graph_dicts(article_df_before)
    id_to_food_dict_aft, food_to_id_dict_aft, id_to_food_count_aft = create_graph_dicts(article_df_after)


# create recipe to recipe network
mini_example_df = pd.read_csv('mini_example.csv')
mini_net = g = nx.from_pandas_edgelist(mini_example_df,
                             source='source',
                            target='target',
                           edge_attr='weight')
# network_recipe_bef_df, network_recipe_bef = get_df_shared_ingredients_network(id_to_food_dict_bef)
# network_recipe_aft = get_df_shared_ingredients_network(id_to_food_dict_aft)

# get subnets
ex_subs = get_subnets(mini_net)
print(len(ex_subs))
for sub in ex_subs:
    print(len(sub))
    visualize_networks(sub)
# visualize networks
# visualize_networks(network_recipe_bef)
# visualize_networks(network_recipe_aft)

if False:
    # create distance matrices
    dist_dict_bef = create_distance_matrix(id_to_food_dict_bef, food_to_id_dict_bef, id_to_food_count_bef)
    dist_dict_aft = create_distance_matrix(id_to_food_dict_aft, food_to_id_dict_aft, id_to_food_count_aft)
    # collapse into 1d
    one_dim_bef, id_list_bef = convert_to_1d(dist_dict_bef)
    one_dim_aft, id_list_aft = convert_to_1d(dist_dict_aft)

    pickle.dump(one_dim_bef, open('one_dim_bef.pkl', "wb"))
    pickle.dump(one_dim_aft, open('one_dim_aft.pkl', "wb"))

    pickle.dump(id_list_bef, open('id_list_bef.pkl', "wb"))
    pickle.dump(id_list_aft, open('id_list_aft.pkl', "wb"))

    pickle.dump(dist_dict_bef, open('dist_dict_bef.pkl', "wb"))
    pickle.dump(dist_dict_aft, open('dist_dict_aft.pkl', "wb"))

    # create linkages
    linkage_bef = linkage(one_dim_bef, 'single')
    linkage_aft = linkage(one_dim_aft, 'single')
    pickle.dump(linkage_bef, open("linkage_bef", "wb"))
    pickle.dump(linkage_aft, open("linkage_aft", "wb"))


    # get max heights
    max_height_bef = max(set(one_dim_bef))
    max_height_aft = max(set(one_dim_aft))

    # create clusters
    cluster_range = 8
    cluster_list_bef = []
    cluster_list_aft = []
    for i in range(2, cluster_range):
        print('num cluster: ', i)
        cluster_bef = findCluster(i, linkage_bef, 300, len(id_list_bef))
        cluster_aft = findCluster(i, linkage_aft, 300, len(id_list_aft))
        cluster_list_bef.append(cluster_bef)
        cluster_list_aft.append(cluster_aft)

    # get cluster silhouette score dict/graph
    score_dict_bef, optimal_cluster_bef = find_optimal_clusters(cluster_list_bef, dist_dict_bef,
                                                                id_list_bef, 'cluster_scores_before.jpg')
    score_dict_aft, optimal_cluster_aft = find_optimal_clusters(cluster_list_aft, dist_dict_aft, id_list_aft,
                                                                'cluster_scores_after.jpg')

    print(score_dict_bef)
    print(score_dict_aft)

    convert_cluster_list_to_dfs(article_df_before_samp, optimal_cluster_bef, id_list_bef, 'before')
    convert_cluster_list_to_dfs(article_df_after_samp, optimal_cluster_aft, id_list_aft, 'after')






# my_points = [[1,1,0],
#              [0,0,0],
#              [1,0,0],
#              [0.25, 0.25, 0.25]]
# dist_dict = {'1': {'1': 2, '2': 3, '3': 0.2, '4': 0.1},
#              '2': {'1': 1, '2': 3, '3': 0.2, '4': 0.1},
#              '3': {'1': 0, '2': 2, '3': 0.2, '4': 0.1},
#              '4': {'1': 0.1, '2': 3, '3': 0.4, '4': 1}}
# distance_list, id_list = convert_to_1d(dist_dict)
# my_linkage = linkage(my_points)
#
# plt.figure(figsize=(10, 7))
# dendrogram(my_linkage,
#             orientation='top',
#             distance_sort='descending',
#             show_leaf_counts=True)
# plt.show()

# max_height = math.sqrt(2)
#
# print(my_linkage)

# one_cluster = findCluster(1, my_linkage, max_height, 4)
# two_cluster = findCluster(2, my_linkage, max_height, 4)
# three_cluster = findCluster(3, my_linkage, max_height, 4)
# four_cluster = findCluster(4, my_linkage, max_height, 4)

# cluster_list = [one_cluster, two_cluster, three_cluster, four_cluster]
# score_dict = find_optimal_clusters(cluster_list, dist_dict, id_list)

# print(one_cluster)
# print(two_cluster)
# print(three_cluster)
# print(four_cluster)
# print(score_dict)


