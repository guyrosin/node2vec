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

neo4j = py2neo.Graph(user='neo4j', password='123')

G = nx.Graph()

start = time.time()
cursor = neo4j.run("MATCH (p:Page) RETURN p.title AS title LIMIT 1000000")
end = time.time()
print(end - start)
exit()
while cursor.forward():
    print(cursor)
