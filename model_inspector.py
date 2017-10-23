import pickle

from gensim.models.keyedvectors import KeyedVectors

word_vectors = KeyedVectors.load_word2vec_format('data/wiki_walksnum4_walklength30.model')

with open('data/node_id_to_name.db', 'rb') as f:
    node_id_to_name = pickle.load(f)

# print(word_vectors.similarity('donald_trump_nnp', 'businessman'))
while True:
    query = input("Enter your query: ").split()
    query = [node_id_to_name[q] for q in query]
    if len(query) == 1:
        try:
            print('most similar words:', word_vectors.most_similar(positive=query))
        except Exception:
            print("word doesn't exist!")
    if len(query) == 2:
        try:
            print('similarity:', word_vectors.similarity(query[0], query[1]))
        except Exception:
            print("word doesn't exist!")
