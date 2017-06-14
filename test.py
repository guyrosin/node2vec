import logging
import networkx as nx
from py2neo import Graph
import unidecode

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                    filename='log.log', filemode='w'
                    )

neo4j_mark_events = Graph(user='neo4j', password='123')

results = neo4j_mark_events.data("MATCH p = ()-[]-() RETURN p")
