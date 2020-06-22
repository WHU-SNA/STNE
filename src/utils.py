import argparse
import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from texttable import Texttable
from tqdm import tqdm
import random as rd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def parameter_parser():
    """
    Parse up command line parameters.
    """
    parser = argparse.ArgumentParser(description="Run SLF.")
    parser.add_argument("--edge-path",
                        nargs="?",
                        default="./input/WikiElec.txt",
                        help="Edge list in txt format.")
    parser.add_argument("--outward-embedding-path",
                        nargs="?",
                        default="./output/WikiElec_outward",
                        help="Outward embedding path.")
    parser.add_argument("--inward-embedding-path",
                        nargs="?",
                        default="./output/WikiElec_inward",
                        help="Inward embedding path.")
    parser.add_argument("--dim",
                        type=int,
                        default=48,
                        help="Dimension of latent factor vector. Default is 48.")
    parser.add_argument("--n",
                        type=int,
                        default=10,
                        help="Number of noise samples. Default is 5.")
    parser.add_argument("--window_size",
                        type=int,
                        default=5,
                        help="Context window size. Default is 5.")
    parser.add_argument("--num_walks",
                        type=int,
                        default=20,
                        help="Walks per node. Default is 20.")
    parser.add_argument("--walk_len",
                        type=int,
                        default=10,
                        help="Length per walk. Default is 10.")
    parser.add_argument("--workers",
                        type=int,
                        default=4,
                        help="Number of threads used for random walking. Default is 4.")
    parser.add_argument("--test-size",
                        type=float,
                        default=0.2,
                        help="Test ratio. Default is 0.2.")
    parser.add_argument("--split-seed",
                        type=int,
                        default=16,
                        help="Random seed for splitting dataset. Default is 16.")
    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.02,
                        help="Learning rate. Default is 0.025.")
    parser.add_argument("--m",
                        type=float,
                        default=1,
                        help="Damping factor. Default is 1.")
    parser.add_argument("--norm",
                        type=float,
                        default=0.01,
                        help="Normalization factor. Default is 0.01.")

    return parser.parse_args()


def read_edge_list(args):
    """
    Load edges from a txt file.
    """
    G = nx.DiGraph()
    edges = np.loadtxt(args.edge_path)
    for i in range(edges.shape[0]):
        G.add_edge(int(edges[i][0]), int(edges[i][1]), weight=edges[i][2])
    edges = [[e[0], e[1], e[2]['weight']] for e in G.edges.data()]
    return edges, max(G.nodes) + 1  # index can start from 0.


def parallel_generate_walks(d_graph, walk_len, num_walks, cpu_num):
    """
    Generates the random walks which will be used as the skip-gram input.
    :return: List of walks. Each walk is a list of nodes.
    """

    walks = list()
    pbar = tqdm(total=num_walks, desc='Generating walks (CPU: {})'.format(cpu_num))

    for n_walk in range(num_walks):

        pbar.update(1)

        # Shuffle the nodes
        shuffled_nodes = list(d_graph.keys())
        rd.shuffle(shuffled_nodes)

        # Start a random walk from every node
        for source in shuffled_nodes:

            # Start walk
            walk = [source]

            # Perform walk
            while len(walk) < walk_len:

                walk_options = d_graph[walk[-1]]['successors']
                # Skip dead end nodes
                if not walk_options:
                    break

                probabilities = d_graph[walk[-1]]['probabilities']
                walk_to = np.random.choice(walk_options, size=1, p=probabilities)[0]
                walk.append(walk_to)

            # walk with length < 3 does not work in optimization
            if len(walk) > 2:
                walks.append(walk)

    pbar.close()

    return walks


@ignore_warnings(category=ConvergenceWarning)
def sign_prediction(out_emb, in_emb, train_edges, test_edges):
    """
    Evaluate the performance on the sign prediction task.
    :param out_emb: Outward embeddings.
    :param in_emb: Inward embeddings.
    :param train_edges: Edges for training the model.
    :param test_edges: Edges for test.
    """
    out_dim = out_emb.shape[1]
    in_dim = in_emb.shape[1]
    train_edges = train_edges
    train_x = np.zeros((len(train_edges), (out_dim + in_dim) * 2))
    train_y = np.zeros((len(train_edges), 1))
    for i, edge in enumerate(train_edges):
        u = edge[0]
        v = edge[1]
        if edge[2] > 0:
            train_y[i] = 1
        else:
            train_y[i] = 0
        train_x[i, : out_dim] = out_emb[u]
        train_x[i, out_dim: out_dim + in_dim] = in_emb[u]
        train_x[i, out_dim + in_dim: out_dim * 2 + in_dim] = out_emb[v]
        train_x[i, out_dim * 2 + in_dim:] = in_emb[v]

    test_edges = test_edges
    test_x = np.zeros((len(test_edges), (out_dim + in_dim) * 2))
    test_y = np.zeros((len(test_edges), 1))
    for i, edge in enumerate(test_edges):
        u = edge[0]
        v = edge[1]
        if edge[2] > 0:
            test_y[i] = 1
        else:
            test_y[i] = 0
        test_x[i, : out_dim] = out_emb[u]
        test_x[i, out_dim: out_dim + in_dim] = in_emb[u]
        test_x[i, out_dim + in_dim: out_dim * 2 + in_dim] = out_emb[v]
        test_x[i, out_dim * 2 + in_dim:] = in_emb[v]

    ss = StandardScaler()
    train_x = ss.fit_transform(train_x)
    test_x = ss.fit_transform(test_x)
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(train_x, train_y.ravel())
    test_y_score = lr.predict_proba(test_x)[:, 1]
    test_y_pred = lr.predict(test_x)
    auc_score = roc_auc_score(test_y, test_y_score, average='macro')
    macro_f1_score = f1_score(test_y, test_y_pred, average='macro')

    return auc_score, macro_f1_score


def args_printer(args):
    """
    Print the parameters in tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    t = Texttable()
    l = [[k, args[k]] for k in args.keys()]
    l.insert(0, ["Parameter", "Value"])
    t.add_rows(l)
    print(t.draw())



