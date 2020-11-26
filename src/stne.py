"""
    Social Trust Network Embedding (STNE)
"""

import networkx as nx
import random
import numpy as np
from collections import defaultdict
from utils import sigmoid, read_edge_list, parallel_generate_walks, sign_prediction
from sklearn.model_selection import train_test_split
from motif_count import Motif
from joblib import Parallel, delayed
import random as rd
from tqdm import tqdm


class STNE(object):
    def __init__(self, args):
        self.args = args
        self.logs = {'epoch': [], 'sign_prediction_auc': [], 'sign_prediction_macro_f1': []}
        self.edges, self.num_nodes, self.train_edges, self.test_edges, self.G, self.d_graph = self._setup()

        # Outward latent factor vectors.
        self.out_lf = np.matrix(np.random.rand(self.num_nodes, self.args.dim), dtype=np.float32)
        # Inward latent factor vectors.
        self.in_lf = np.matrix(np.random.rand(self.num_nodes, self.args.dim), dtype=np.float32)
        # Outward trust pattern vectors.
        self.out_tp = np.matrix(np.zeros((self.num_nodes, 16)), dtype=np.float32)
        # Inward trust pattern vectors.
        self.in_tp = np.matrix(np.zeros((self.num_nodes, 16)), dtype=np.float32)

        self.Motif = Motif(self.train_edges)
        self.walks = self._generate_walks()

    def _setup(self):
        edges, num_nodes = read_edge_list(self.args)
        train_edges, test_edges = train_test_split(edges,
                                                   test_size=self.args.test_size,
                                                   random_state=self.args.split_seed)
        d_graph = defaultdict(dict)
        G = nx.DiGraph()
        for edge in train_edges:
            if edge[2] > 0:
                G.add_edge(edge[0], edge[1], weight=edge[2], polarity=1)
            elif edge[2] < 0:
                G.add_edge(edge[0], edge[1], weight=abs(edge[2]), polarity=-1)
        for node in G.nodes():
            unnormalized_weights = []
            succs = list(G.successors(node))

            if not succs:
                d_graph[node]['probabilities'] = []
                d_graph[node]['successors'] = []
            else:
                for succ in succs:
                    weight = G[node][succ]['weight']
                    unnormalized_weights.append(weight)
                unnormalized_weights = np.array(unnormalized_weights)
                d_graph[node]['probabilities'] = unnormalized_weights / unnormalized_weights.sum()
                d_graph[node]['successors'] = succs

        return edges, num_nodes, train_edges, test_edges, G, d_graph

    def _generate_walks(self):
        flatten = lambda l: [item for sublist in l for item in sublist]
        num_walks_lists = np.array_split(range(self.args.num_walks), self.args.workers)

        walk_results = Parallel(n_jobs=self.args.workers)(
            delayed(parallel_generate_walks)(self.d_graph,
                                             self.args.walk_len,
                                             len(num_walks),
                                             idx, ) for
            idx, num_walks in enumerate(num_walks_lists))

        return flatten(walk_results)

    def _calculate_negative_sampling_distribution(self):
        sampling_dis = {}
        nodes = list(self.d_graph.keys())
        for node in nodes:
            sampling_dis[node] = {}
        for walk in self.walks:
            walk_len = len(walk)
            for start in range(walk_len - 1):
                u = walk[start]
                sign = 1
                context = walk[start + 1: min(start + self.args.window_size + 1, self.args.walk_len)]
                pre_v = u
                path_len = 0
                for v in context:
                    if v == u:
                        break
                    sign *= self.G[pre_v][v]['polarity']
                    trust = ((self.args.window_size - path_len) / self.args.window_size) ** self.args.m
                    score = sign * trust
                    if v not in sampling_dis[u].keys():
                        sampling_dis[u][v] = score
                    else:
                        sampling_dis[u][v] += score
                    path_len += 1
                    pre_v = v
        pos_dis = {}
        neg_dis = {}
        for node in nodes:
            pos_dis[node] = {}
            neg_dis[node] = {}
        for node in sampling_dis.keys():
            for target in sampling_dis[node].keys():
                score = sampling_dis[node][target]
                if score > 0:
                    pos_dis[node][target] = score
                elif score < 0:
                    neg_dis[node][target] = abs(score)
        return pos_dis, neg_dis

    def fit(self):
        pbar = tqdm(total=len(self.walks), desc='Optimizing', ncols=100)
        pos_dis, neg_dis = self._calculate_negative_sampling_distribution()
        nodes = list(self.d_graph.keys())
        rd.shuffle(self.walks)
        for walk in self.walks:
            pbar.update(1)
            walk_len = len(walk)
            for start in range(walk_len - 1):
                u = walk[start]
                sign = 1
                context = walk[start + 1: min(start + self.args.window_size + 1, self.args.walk_len)]
                pre_v = u
                path_len = 0
                for v in context:
                    if v == u:
                        break
                    sign *= self.G[pre_v][v]['polarity']
                    trust = ((self.args.window_size - path_len) / self.args.window_size) ** self.args.m
                    path_len += 1

                    triad_uv = np.array(self.Motif.motif_vector(u, v))
                    X = self.out_lf[u] @ self.in_lf[v].T + (1 / 2) * (self.out_tp[u] + self.in_tp[v]) @ triad_uv.T
                    p_uv = sigmoid(sign * X)
                    out_u_g = 0
                    out_u_g += (sign * trust * (1 - p_uv)) * self.in_lf[v]
                    self.in_lf[v] += self.args.learning_rate * ((sign * trust * (1 - p_uv)) * self.out_lf[u] - self.args.norm * self.in_lf[v])
                    self.out_tp[u] += self.args.learning_rate * ((sign * trust * (1 - p_uv) / 2) * triad_uv - self.args.norm * self.out_tp[u])
                    self.in_tp[v] += self.args.learning_rate * ((sign * trust * (1 - p_uv) / 2) * triad_uv - self.args.norm * self.in_tp[v])
                    pre_v = v

                    # negative sampling
                    if sign > 0:
                        dis = neg_dis[u]
                    else:
                        dis = pos_dis[u]
                    if len(dis.keys()) == 0:
                        noises = random.choices(nodes, k=self.args.n)
                    else:
                        noises = random.choices(list(dis.keys()), list(dis.values()), k=self.args.n)
                    for noise in noises:
                        X = self.out_lf[u] @ self.in_lf[noise].T
                        p_unoise = sigmoid(-sign * X)
                        out_u_g += (-sign * trust * (1 - p_unoise)) * self.in_lf[noise]
                        self.in_lf[noise] += self.args.learning_rate * (
                                    (-sign * trust * (1 - p_unoise)) * self.out_lf[u] - self.args.norm * self.in_lf[noise])
                    self.out_lf[u] += self.args.learning_rate * (out_u_g - self.args.norm * self.out_lf[u])
        pbar.close()
        W_out = np.matrix(np.zeros((self.num_nodes, self.args.dim + 16)), dtype=np.float32)
        W_in = np.matrix(np.zeros((self.num_nodes, self.args.dim + 16)), dtype=np.float32)
        for i in range(self.num_nodes):
            W_out[i, : self.args.dim] = self.out_lf[i]
            W_out[i, self.args.dim:] = self.out_tp[i]
            W_in[i, : self.args.dim] = self.in_lf[i]
            W_in[i, self.args.dim:] = self.in_tp[i]
        auc, f1 = sign_prediction(W_out, W_in, self.train_edges, self.test_edges)
        print('Sign prediction: AUC %.3f, F1 %.3f' % (auc, f1))

    def save_emb(self):
        """
        Save the node embeddings in .npz format.
        """
        W_out = np.matrix(np.zeros((self.num_nodes, self.args.dim + 16)), dtype=np.float32)
        W_in = np.matrix(np.zeros((self.num_nodes, self.args.dim + 16)), dtype=np.float32)
        for i in range(self.num_nodes):
            W_out[i, : self.args.dim] = self.out_lf[i]
            W_out[i, self.args.dim:] = self.out_tp[i]
            W_in[i, : self.args.dim] = self.in_lf[i]
            W_in[i, self.args.dim:] = self.in_tp[i]
        np.save(self.args.outward_embedding_path, W_out)
        np.save(self.args.inward_embedding_path, W_in)
