import numpy as np
class VectorStorage:
    def __init__(self):
             self.vector_data = {} # empty dictionary to store vector data
             self.vector_index = {} # empty dictionary to store vector index for indexing


    def add_vector(self, vector_id, vector):
           """Add a vector to the vector store

           Args:
               vector_id (str or int): A unique identifier for the vector
               vector (numpy.ndarray): The vector to be added
           """
           self.vector_data[vector_id] = vector
           self._update_index(vector_id, vector)


    def get_vector(self, vector_id):
           """Get a vector from the vector store

           Args:
               vector_id (str): A unique identifier for the vector
           Returns:
               numpy.ndarray: The vector data if found else None    
           """
           return self.vector_data[vector_id]

    def _update_index(self, vector_id, vector):
           """Update the indexing structure of the vector store 

           Args:
               vector_id (str or int): A unique identifier for the vector
               vector (numpy.ndarray): The vector to be added
           """

           for existing_vector_id, existing_vector in self.vector_data.items():
                  similarity = np.dot(vector, existing_vector) / (np.linalg.norm(vector) * np.linalg.norm(existing_vector))
                  if existing_vector_id not in self.vector_index:
                         self.vector_index[existing_vector_id] = {}

                  self.vector_index[existing_vector_id][vector_id] = similarity


    def get_similar_vectors(self, query_vector, num_results=5):
           """Get similar vectors from the vector store

           Args:
               query_vector (numpy.ndarray): The query vector
               num_results (int): The number of similar vectors to return
           Returns:
               list: A list of similar vectors o the format (vector_id, similarity_score)
           """
           results = []
           for vector_id, vector in self.vector_data.items(): 
                  similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector)) 
                  results.append((vector_id, similarity))
           
           #sort the similarity in descending order
           results.sort(key=lambda x: x[1], reverse=True)
           #return the top N results
           return results[:num_results] 

           
                        

               

        






