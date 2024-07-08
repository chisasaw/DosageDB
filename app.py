import numpy as np
import vector_store 

#create vector store instance
vector_storage = vector_store.VectorStorage()


#define your sentence here in a list
sentences = [
            "england is good at football",
            "football is the best sport",
            "england is good at the best sport",
            "football, basketball, and handball are all sports"
            ]
#Tokenization and vocabulary building
vocabulary = set()

for sentence in sentences:
    tokens = sentence.lower().split()
    vocabulary.update(tokens)

#Assign unique indices to words in the vocabulary
word_to_index = {word: index for index, word in enumerate(vocabulary)}

#Vectorization
sentence_vectors = {}
for sentence in sentences:
    tokens = sentence.lower().split()
    vector = np.zeros(len(vocabulary))
    for token in tokens:
        vector[word_to_index[token]] += 1

    sentence_vectors[sentence] = vector 

#Store the vectors in the vector store
for sentence, vector in sentence_vectors.items():
    vector_storage.add_vector(sentence, vector)

#seaching for similarity
query_sentence = "football is popular in england"
query_vector = np.zeros(len(vocabulary))
query_tokens = query_sentence.lower().split()

for token in query_tokens:
    if token in word_to_index:
        query_vector[word_to_index[token]] += 1

similar_sentences = vector_storage.get_similar_vectors(query_vector, num_results=2)

print("Query sentence: ", query_sentence)
print("Similar sentences: ")

for sentence, similarity in similar_sentences:
    print(f"{sentence}: Similarity ={similarity: 0.4f}")




