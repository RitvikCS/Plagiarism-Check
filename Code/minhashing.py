"""
In this we calculate the signature of each document using minhashing 
technique by utilizing a number of hash functions
"""

from random import randint
from pandas import DataFrame, read_pickle
import numpy as np
import os
from tqdm import tqdm


def gen_hash_func(rows, no_of_hash_functions=200):
    """This function creates parameters for given no of hash functions

    Parameters
    ----------
    rows: Number of rows in shingle matrix (int)
    no_of_hash_functions: Number of hash functions to generate for minhashing (Default: 200)

    Returns
    -------
    list : list of functions which can be used as hashes[i](x)
    """

    hashes = []
    c = rows

    for i in range(no_of_hash_functions):
        def hash(x):
            """
            This function calculates hash for given x
            hash function format: (a*x+b)%c where
                c: prime integer just greater than rows
                a,b: random integer less than c
            """
            return (randint(1, 5*c)*x + randint(1, 5*c)) % c
        hashes.append(hash)

    return hashes


def generate_signature_matrix(inci_mat, no_of_hash_functions=200):
    """It generates the signature matrix for whole corpus

    Parameters
    ----------
    inci_mat: incidence index generated after shingling of similar process(pandas.DataFrame)
    no_of_hash_functions: numner of hash functions to be used in generating signatures for documents.(Default: 100)

    Returns
    -------
    returns dataframe containing signatures of every document
    """

    # if pickle file exists, load and return it
    if os.path.exists("sig_mat.pickle"):
        signature_matrix = read_pickle("sig_mat.pickle")
        print("Using already created sig_mat.pickle file")
        return signature_matrix

    rows, cols = inci_mat.shape
    hashes = gen_hash_func(rows, no_of_hash_functions)
    signature_matrix = DataFrame(index=[i for i in range(
        no_of_hash_functions)], columns=inci_mat.columns)

    # core minhashing algorithm
    for i in tqdm(range(rows)):
        for j in inci_mat.columns:
            if inci_mat.iat[i, j] == 1:
                for k in range(no_of_hash_functions):
                    if np.isnan(signature_matrix.iat[k, j]):
                        signature_matrix.iat[k, j] = hashes[k](i)
                    else:
                        signature_matrix.iat[k, j] = min(
                            signature_matrix.iat[k, j], hashes[k](i))

    print("Saving generated signature_matrix to pickle file...")
    signature_matrix.to_pickle("sig_mat.pickle")
    print("Saved to sig_mat.pickle")
    return signature_matrix
