import argparse

import graph_tool
import time
import pickle
import os

import node2vec
from gensim.models import Word2Vec


def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='wiki.gt',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='emb/karate.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=60,  # was 80
                        help='Length of walk per source. Default is 60.')

    parser.add_argument('--num-walks', type=int, default=8,  # was 10
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=5,  # was 10
                        help='Context size for optimization. Default is 5.')

    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--min-count', type=int, default=5,  # wasn't here originally (w2v's default is 5)
                        help='Minimum count for word2vec. Default is 5.')

    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers. Default is 4.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is directed.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=True)

    return parser.parse_args()


def read_graph(graph_file_path):
    '''
    Reads the input network in graph_tool, as a directed graph
    '''
    start = time.time()
    g = graph_tool.load_graph(graph_file_path)
    end = time.time()
    print('loading the graph took {}'.format(end - start))
    return g


def learn_embeddings(walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [map(str, walk) for walk in walks]
    start = time.time()
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=args.min_count, sg=1,
                     workers=args.workers, iter=args.iter)
    end = time.time()
    print('building th w2v model took {}'.format(end - start))
    model.save_word2vec_format(args.output)

    return


def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    walks_file_path = 'walks.pickle'
    alias_nodes_file_path = 'alias_nodes.pickle'
    alias_edges_file_path = 'alias_edges.pickle'
    if not os.path.isfile(walks_file_path):
        gt_g = read_graph(args.input)
        G = node2vec.Graph(gt_g, args.directed, args.p, args.q)
        if not os.path.isfile(alias_nodes_file_path) or not os.path.isfile(alias_edges_file_path):
            start = time.time()
            G.preprocess_transition_probs()
            end = time.time()
            print('preprocessing probs took {}'.format(end - start))
        else:
            with open(alias_nodes_file_path, 'rb') as file_obj:
                G.alias_nodes = pickle.load(file_obj)
            with open(alias_edges_file_path, 'rb') as file_obj:
                G.alias_edges = pickle.load(file_obj)
        start = time.time()
        walks = G.simulate_walks(args.num_walks, args.walk_length)
        end = time.time()
        print('simulating walks took {}'.format(end - start))
        with open(walks_file_path, 'wb') as file_obj:
            pickle.dump(walks, file_obj)
    else:
        with open(walks_file_path, 'rb') as file_obj:
            walks = pickle.load(file_obj)
    learn_embeddings(walks)


if __name__ == "__main__":
    args = parse_args()
    main(args)
