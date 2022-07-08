import pandas as pd
import math
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import numpy as np

# constant used for location of scipy linkage matrix
LEFT_INDEX = 0
RIGHT_INDEX = 1
HEIGHT_INDEX = 2
NUM_ELEMENTS_INDEX = 3



# # create sparse vectors (counts?)
article_df = pd.read_csv('cleaned_stratified.csv')
def create_graph_dicts(article_df):
    story_ids = set(list(article_df['story_id']))
    id_to_food_dict = {}
    food_to_id_dict = {}
    id_to_food_count = {}
    for story_id in story_ids:
        food_df = article_df[article_df['story_id'] == story_id]
        food_list = list(food_df['text'])
        if len(food_list) == 0:
            continue
        id_to_food_dict[story_id] = food_list
        id_to_food_count[story_id] = len(food_list)
        for food in food_list:
            try:
                food_to_id_dict[food].append(story_id)
            except KeyError:
                food_to_id_dict[food] = [story_id]
    return id_to_food_dict, food_to_id_dict, id_to_food_count
#
# vectorizer = CountVectorizer(analyzer=lambda x: x)
# vectorize_fit = vectorizer.fit(food_lists)
# X = vectorizer.transform(food_lists).toarray()
# print('ingredient count')
# print(X.shape)
# # kmeans clustering?
#
# # from sklearn.cluster import KMeans
# # import numpy as np
# # kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
# # print(kmeans.labels_)
#
# # what is a dendrogram?
# from scipy.cluster.hierarchy import dendrogram, linkage
# from matplotlib import pyplot as plt
#
#
# #
# # linked = linkage(X, 'single')
# # print(linked)
# #
# # plt.figure(figsize=(10, 7))
# # dendrogram(linked,
# #             orientation='top',
# #             distance_sort='descending',
# #             show_leaf_counts=True)
# # plt.show()
#
# # from sklearn.cluster import AgglomerativeClustering
# #
# # cluster = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='ward',
# #                                   distance_threshold=2)
# # cluster.fit_predict(X)
# # plt.scatter(X[:,0],X[:,1], c=cluster.labels_, cmap='rainbow')
# # plt.show()
#
#
# # Create graph with the nodes as recipes and ingredients with an edge for each
# # ingredient contained in the recipe
#
# # Initial dictionary: article id/other media id -> ingredients (list of ingredients lowercased)
# # this is id_to_food_dict
#
# # Second Dictionary: ingredient name -> article id/other media id
# # this is food_to_id_dict
#

import networkx as nx
from pyvis.network import Network

def visualize_networks(network):
    # g = nx.from_pandas_edgelist(network_dict,
    #                             source='source',
    #                             target='target',
    #                             edge_attr='weight')
    nx.draw(network, with_labels=True)
    plt.show()
    # net = Network(notebook = True)
    # net.from_nx(g)
    # net.show('network_a.html')

def divide_unconnected_nodes(network):
    # divide the network by unconnected nodes
    pass

def cluster_network(network):
    comp = nx.girvan_newman(network)
    comms = tuple(sorted(c) for c in next(comp))
    print(comms)
    return comms


def get_furthest_nodes(G):
    sp_length = {}  # dict containing shortest path distances for each pair of nodes
    diameter = None  # will contain the graphs diameter (length of longest shortest path)
    furthest_node_list = []  # will contain list of tuple of nodes with shortest path equal to diameter

    for node in G.nodes:
        # Get the shortest path from node to all other nodes
        sp_length[node] = nx.single_source_shortest_path_length(G, node)
        longest_path = max(sp_length[node].values())  # get length of furthest node from node

        # Update diameter when necessary (on first iteration and when we find a longer one)
        if diameter == None:
            diameter = longest_path  # set the first diameter

        # update the list of tuples of furthest nodes if we have a best diameter
        if longest_path >= diameter:
            diameter = longest_path

            # a list of tuples containing
            # the current node and the nodes furthest from it
            node_longest_paths = [(node, other_node)
                                  for other_node in sp_length[node].keys()
                                  if sp_length[node][other_node] == longest_path]
            if longest_path > diameter:
                # This is better than the previous diameter
                # so replace the list of tuples of diameter nodes with this nodes
                # tuple of furthest nodes
                furthest_node_list = node_longest_paths
            else:  # this is equal to the current diameter
                # add this nodes tuple of furthest nodes to the current list
                furthest_node_list = furthest_node_list + node_longest_paths

    # return the diameter,
    # all pairs of nodes with shortest path length equal to the diameter
    # the dict of all-node shortest paths
    return furthest_node_list

def get_edges_to_cut(network):
    # find diameter
    furthest_nodes = get_furthest_nodes(network)
    print(furthest_nodes)
    _, edges = nx.minimum_cut(network, furthest_nodes[-1][0], furthest_nodes[-1][1])
    edges_to_cut = get_cutset(edges, network)
    return edges_to_cut

def get_cutset(edges_to_cut, network):
    reachable, non_reachable = edges_to_cut
    cutset = set()
    for u, nbrs in ((n, network[n]) for n in reachable):
        cutset.update((u, v) for v in nbrs if v in non_reachable)
    return cutset

def get_subnets(network):
    graph_list = [network.subgraph(c).copy() for c in nx.connected_components(network)]
    # the number of edges in the min cut is greater than
    # the number of vertices divided by 2
    list_of_clusters = []
    print('graph list', graph_list)
    while len(graph_list) > 0:
        graph = graph_list.pop(0)
        print('graph list', graph_list)
        min_edge_cut = get_edges_to_cut(graph)
        print(min_edge_cut)
        num_vertices = graph.number_of_nodes()
        if len(min_edge_cut) > (num_vertices / 2):
            list_of_clusters.append(graph)
        else:

            for edge in min_edge_cut:
                graph.remove_edge(edge[0], edge[1])
            comps = nx.connected_components(graph)
            print(comps)
            for comp in comps:
                print(comp)
                graph_list.append(graph.subgraph(comp).copy())
        raise ValueError('stupid test')
    return list_of_clusters

def get_df_shared_ingredients_network(info_dict):
    graph_type = 'undirected'
    graph_dict_list = []
    start_index = 0
    key_list = list(info_dict.keys())
    for story_id_index in range(len(key_list)):
        source = key_list[story_id_index]
        source_list = info_dict[source]
        for story_id_index2 in range(len(key_list)):
            target = key_list[story_id_index2]
            target_list = info_dict[target]
            # print(source_list)
            # print(target_list)
            weight = len(np.intersect1d(source_list, target_list))
            # print(weight)
            graph_dict = {'source': source, 'target': target, 'weight': weight, 'type': graph_type}
            graph_dict_list.append(graph_dict)
        start_index += 1
    network_df = pd.DataFrame(graph_dict_list)
    g = nx.from_pandas_edgelist(network_df,
                                source='source',
                                target='target',
                                edge_attr='weight')
    return network_df, g

def garbage(current_node_list, level, visited_list, distance_dict, current_dist_add,
            id_to_food_dict, food_to_id_dict):
    new_node_list = []
    print('level', level)
    if level % 2 == 0:
        print('id keys')
        dict_to_use = id_to_food_dict
    else:
        print('food keys')
        dict_to_use = food_to_id_dict
    for node in current_node_list:
        neighbors = dict_to_use[node]
        for neighbor in neighbors:
            if neighbor not in visited_list:
                new_node_list.append(neighbor)
                visited_list.add(neighbor)
                if level % 2 == 1:
                    distance_dict[neighbor] += current_dist_add
    level += 1
    return new_node_list, level

# Distance Matrix Dictionary of Dictionaries with the media id's as keys for each
def create_distance_matrix(id_to_food_dict, food_to_id_dict, id_to_food_count):
    distance_matrix = {}
    story_ids = id_to_food_dict.keys()
    for story_id, food_list in id_to_food_dict.items():
        # create a dictionary with the keys being each of the other
        # story_ids and the value being the calculated distance
        current_story_dist_dict = {}
        for st_id in story_ids:
            current_story_dist_dict[st_id] = 0
        level = 0
        # go through the list of connected story ids, and find the next
        # level of connected story ids by iterating through each list of food
        # that has not been visited yet
        # ends loop when there are no more unvisited stories and no more food in the list
        # or no more unvisited food and no more stories in the list of connections
        visited_set = {story_id}
        current_node_list = [story_id]
        while len(current_node_list) > 0:
            current_dist_addition = 1 * pow(math.e, -level)
            current_node_list, level = garbage(current_node_list, level, visited_set,
                                               current_story_dist_dict, current_dist_addition,
                                               id_to_food_dict, food_to_id_dict)

        # normalize for the number of ingredients in the two recipes
        current_story_food_count = id_to_food_count[story_id]
        for curr_story_id in current_story_dist_dict.keys():
            total_food_count = current_story_food_count + id_to_food_count[curr_story_id]
            raw_dist = current_story_dist_dict[curr_story_id]
            if raw_dist == 0:
                current_story_dist_dict[curr_story_id] = 100000
            else:
                current_story_dist_dict[curr_story_id] = pow((raw_dist / total_food_count), -1)
        # add current story distance list to matrix
        distance_matrix[story_id] = current_story_dist_dict
    return distance_matrix

# # Look for every path from the recipe to any other recipes and fill in the values
#
# # Breadth first search through the graph and add (1 * e ^ x) to every recipe encountered
# # where x is the level of intersection (the level of distance)
#
# # Create the distance matrix by dividing the distance found between each recipe by the sum
# # of the ingredients in both recipes and taking the inverse of that [2, infinity)
#
# # Feed the distance matrix into hierarchical clustering (linkage)
#
def convert_to_1d(distance_dict):
    # list of ids in order
    id_list = list(distance_dict.keys())
    distance_list = []
    start_index = 0

    for ident in id_list:
        for ident2_index in range(start_index, len(id_list)):
            ident2 = id_list[ident2_index]
            dist_to_add = distance_dict[ident][ident2]
            distance_list.append(dist_to_add)
            print(start_index, ident2_index)
        start_index += 1

    return distance_list, id_list



# Iterate through the possible number of clusters by iterating through heights in the dendrogram,
# and calculate the average distance between points in each cluster and average distance to the
# next cluster (using the original distance matrix)

# Graph the calculated values and find the global maximum if it exists - that is the optimal number
# of clusters for the data

def findCluster(goal_number_cluster, heirarcal_info, upper_bound, number_of_total_recipies):
    lower_bound = 0
    while True:
        print('lower: ', lower_bound, ' upper: ', upper_bound)
        middle = (lower_bound + upper_bound) / 2
        # gets the number of clusters at the height of mid
        clusters = get_clusters_at_height(middle, heirarcal_info, number_of_total_recipies)
        current_num_clusters = len(clusters)
        print('current num', current_num_clusters)
        if current_num_clusters < goal_number_cluster:
            print('step left')
            # step left (lower height - more clusters)
            upper_bound = middle - 0.00001
        elif current_num_clusters > goal_number_cluster:
            print('step right')
            # step right (greater height - less clusters)
            lower_bound = middle + 0.00001
        elif lower_bound > upper_bound:
            raise ValueError('no height exists for this number of clusters, or I wrote code wrong')
        else:
            print('should break')
            # the correct number of clusters was found so break loop and return the clusters
            break

    return clusters


def get_clusters_at_height(target_height, heirarcal_info, number_of_total_recipies):
    # starts at highest level cluster
    indecies_of_clusters = [len(heirarcal_info) - 1]
    print('indices of clusters: ', indecies_of_clusters)

    final_clusters_indices = []

    number_of_clusters = 1
    print('number of clusters: ', number_of_clusters)
    while len(indecies_of_clusters) > 0:
        current_index = indecies_of_clusters.pop(0)

        # if the value is negative that means it is a base index (index - n)
        if current_index < 0:
            final_clusters_indices.append(current_index)
            continue

        information_for_index = heirarcal_info[current_index]
        print('info', information_for_index)
        current_height_of_clusters = information_for_index[HEIGHT_INDEX]
        if target_height < current_height_of_clusters:
            print('got here')
            # if the height of this cluster is greater than the target height then it should be split
            left_index = int(information_for_index[LEFT_INDEX]) - number_of_total_recipies
            right_index = int(information_for_index[RIGHT_INDEX]) - number_of_total_recipies
            indecies_of_clusters.append(left_index)
            indecies_of_clusters.append(right_index)
            number_of_clusters += 1
        else:
            # this cluster is one of the final clusters
            print('final')
            final_clusters_indices.append(current_index)

    clusters = []
    print('final clust index', final_clusters_indices)
    for cluster_index in final_clusters_indices:
        cluster = cluster_index_to_cluster(cluster_index, heirarcal_info, number_of_total_recipies)
        clusters.append(cluster)

    return clusters


def cluster_index_to_cluster(cluster_index, heirarcal_info, number_of_total_recipies):
    # used to find indices of all recipies in the cluster
    print('start construction of index')
    cluster = set()
    # does this by traversing through it as a tree then adds any starting nodes seen to a set.
    indices_of_clusters = [cluster_index]
    while len(indices_of_clusters) > 0:
        current_index = indices_of_clusters.pop(0)
        if current_index < 0:
            cluster.add(current_index + number_of_total_recipies)
        else:
            # add the left and right side of the tree
            print('current_index: ', current_index)
            left_index = int(heirarcal_info[current_index][LEFT_INDEX]) - number_of_total_recipies
            right_index = int(heirarcal_info[current_index][RIGHT_INDEX]) - number_of_total_recipies
            indices_of_clusters.append(left_index)
            indices_of_clusters.append(right_index)

    return cluster

def get_avg_distance_clust(cluster_indices, dist_dict, id_list, initial_id):
    # the average distance from a point (initial_id) to all the points
    # in the given cluster (cluster_indices)
    avg_dist = 0
    for s_index in cluster_indices:
        sid = id_list[s_index]
        dist = dist_dict[initial_id][sid]
        avg_dist += dist
    return avg_dist / len(cluster_indices)

def get_silhouette_score_individual(initial_id, cluster_indices, closest_cluster_indices, dist_dict, id_list):
    print('got to individual')
    a_i = get_avg_distance_clust(cluster_indices, dist_dict, id_list, initial_id)
    b_i = get_avg_distance_clust(closest_cluster_indices, dist_dict, id_list, initial_id)
    s_i = (b_i - a_i) / max(b_i, a_i)
    return s_i

def get_silhouette_score_avg(clusters, dist_dict, id_list):
    print('got to average sil score func')
    avg_sil_score = 0
    for cluster_indices_index in range(len(clusters)):
        cluster_indices = clusters[cluster_indices_index]
        for init_id_index in cluster_indices:
            init_id = id_list[init_id_index]
            try:
                closest_cluster = clusters[cluster_indices_index + 1]
            except IndexError:
                closest_cluster = clusters[0]
            s_i = get_silhouette_score_individual(init_id, cluster_indices,
                                                  closest_cluster, dist_dict, id_list)
            avg_sil_score += s_i

    return avg_sil_score / len(dist_dict)

def find_optimal_clusters(clusters_option_list, dist_dict, id_list, name):
    # trying to find the place in the count of clusters where the silhouette scores decrease
    # on either side, so at the increase of cluster count and decrease of cluster count
    # check the left and right and move towards the side that goes up if there is one
    # (if there is not this means this is the ideal num of clusters?)
    # after moving in the direction of up (can cut off any other cluster counts) and then keep
    # repeating until the end
    left_score = 100000
    right_score = 100000
    current_score = 0
    clusters_option_index = int (len(clusters_option_list) / 2)
    score_dict = {}
    # while left_score > current_score or right_score > current_score:
    #     left = clusters_option_list[clusters_option_index - 1]
    #     right = clusters_option_list[clusters_option_index + 1]
    #     current = clusters_option_list[clusters_option_index]
    #     cluster_count = len(current)
    #     current_score = get_silhouette_score_avg(current, dist_dict, id_list)

    current_high_score = -math.inf
    optimal_clusters = clusters_option_list[0]
    for cluster in clusters_option_list:
        cluster_count = len(cluster)
        print(cluster_count)
        current_score = get_silhouette_score_avg(cluster, dist_dict, id_list)
        print(current_score)
        score_dict[cluster_count] = current_score
        if current_score > current_high_score:
            current_high_score = current_score
            optimal_clusters = cluster

    cluster_counts = list(score_dict.keys())
    scores = list(score_dict.values())
    plt.plot(cluster_counts, scores)
    plt.savefig(name)
    return score_dict, optimal_clusters
