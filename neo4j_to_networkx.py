import csv

import py2neo
import networkx as nx
import time


# from pygraphml import GraphMLParser
#
# parser = GraphMLParser()
# start = time.time()
# g = parser.parse("D:\\Research\\node2vec\\graph\\wikilinks.graphml")
# end = time.time()
# print(end - start)
# exit()

def neo4j_remove_titles(neo4j):
    start = time.time()
    neo4j.run("MATCH (p:Page) REMOVE p.title")
    end = time.time()
    print('neo4j query took {}'.format(end - start))


def neo4j_to_nodes_dictionary(neo4j):
    start = time.time()
    cursor = neo4j.run("MATCH (p:Page) RETURN ID(p) as pid, p.title AS title")
    end = time.time()
    print('neo4j query took {}'.format(end - start))
    dict = {}
    start = time.time()
    while cursor.forward():
        dict[cursor.current()['pid']] = cursor.current()['title']
    end = time.time()
    print('building the dictionary took {}'.format(end - start))
    with open('mycsvfile.csv', 'w', newline='', encoding='utf8') as f:
        w = csv.writer(f)
        w.writerows(dict.items())


if __name__ == '__main__':
    neo4j = py2neo.Graph(user='neo4j', password='123')
    neo4j_remove_titles(neo4j)
    # neo4j_to_nodes_dictionary(neo4j)
    # G = nx.Graph()
