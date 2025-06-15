from abc import ABC, abstractmethod

from collections import defaultdict
import os
import pickle

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WordEmbedding(ABC):

    def __init__(self, embedding_matrix, word2index, index2word, nn_matrix=None):
        self.embedding_matrix = embedding_matrix
        self._word2index = word2index
        self._index2word = index2word
        self.nn_matrix = nn_matrix

        # Dictionary for caching results
        self._mse_dist_mat = defaultdict(dict)
        self._cos_sim_mat = defaultdict(dict)
        self._nn_cache = {}

    def __getitem__(self, index):

        if isinstance(index, str):
            try:
                index = self._word2index[index]
            except KeyError:
                return None
        try:
            return self.embedding_matrix[index]
        except IndexError:
            # word embedding ID out of bounds
            return None
        
    def word2index(self, word):

        return self._word2index[word]

    def index2word(self, index):

        return self._index2word[index]
    
    def get_mse_dist(self, a, b):

        if isinstance(a, str):
            a = self._word2index[a]
        if isinstance(b, str):
            b = self._word2index[b]
        a, b = min(a, b), max(a, b)
        try:
            mse_dist = self._mse_dist_mat[a][b]
        except KeyError:
            e1 = self.embedding_matrix[a]
            e2 = self.embedding_matrix[b]
            e1 = torch.tensor(e1).to(device)
            e2 = torch.tensor(e2).to(device)
            mse_dist = torch.sum((e1 - e2) ** 2).item()
            self._mse_dist_mat[a][b] = mse_dist

        return mse_dist

    def get_cos_sim(self, a, b):

        if isinstance(a, str):
            a = self._word2index[a]
        if isinstance(b, str):
            b = self._word2index[b]
        a, b = min(a, b), max(a, b)
        try:
            cos_sim = self._cos_sim_mat[a][b]
        except KeyError:
            e1 = self.embedding_matrix[a]
            e2 = self.embedding_matrix[b]
            e1 = torch.tensor(e1).to(device)
            e2 = torch.tensor(e2).to(device)
            cos_sim = torch.nn.CosineSimilarity(dim=0)(e1, e2).item()
            self._cos_sim_mat[a][b] = cos_sim
        return cos_sim

    def nearest_neighbours(self, index, topn):

        if isinstance(index, str):
            index = self._word2index[index]
        if self.nn_matrix is not None:
            nn = self.nn_matrix[index][1 : (topn + 1)]
        else:
            try:
                nn = self._nn_cache[index]
            except KeyError:
                embedding = torch.tensor(self.embedding_matrix).to(device)
                vector = torch.tensor(self.embedding_matrix[index]).to(device)
                dist = torch.norm(embedding - vector, dim=1, p=None)
                # Since closest neighbour will be the same word, we consider N+1 nearest neighbours
                nn = dist.topk(topn + 1, largest=False)[1:].tolist()
                self._nn_cache[index] = nn

        return nn

    @staticmethod
    def counterfitted_GLOVE_embedding():

        word_embeddings_folder = "paragramcf"
        word_embeddings_file = "paragram.npy"
        word_list_file = "wordlist.pickle"
        mse_dist_file = "mse_dist.p"
        cos_sim_file = "cos_sim.p"
        nn_matrix_file = "nn.npy"

        # Download embeddings if they're not cached.
        word_embeddings_folder = os.path.join("EMBEDDING_WEIGHTS", word_embeddings_folder)
        
        # Concatenate folder names to create full path to files.
        word_embeddings_file = os.path.join(
            word_embeddings_folder, word_embeddings_file
        )
        word_list_file = os.path.join(word_embeddings_folder, word_list_file)
        mse_dist_file = os.path.join(word_embeddings_folder, mse_dist_file)
        cos_sim_file = os.path.join(word_embeddings_folder, cos_sim_file)
        nn_matrix_file = os.path.join(word_embeddings_folder, nn_matrix_file)

        # loading the files
        embedding_matrix = np.load(word_embeddings_file)
        word2index = np.load(word_list_file, allow_pickle=True)
        index2word = {}
        for word, index in word2index.items():
            index2word[index] = word
        nn_matrix = np.load(nn_matrix_file)

        embedding = WordEmbedding(embedding_matrix, word2index, index2word, nn_matrix)

        with open(mse_dist_file, "rb") as f:
            mse_dist_mat = pickle.load(f)
        with open(cos_sim_file, "rb") as f:
            cos_sim_mat = pickle.load(f)

        embedding._mse_dist_mat = mse_dist_mat
        embedding._cos_sim_mat = cos_sim_mat


        return embedding