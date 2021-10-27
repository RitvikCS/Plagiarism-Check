"""
Various metrics which are required to evaluate LSH.
"""
import numpy as np


def jaccard(x, a, signature_matrix):
    """Calculates the Jaccard similarity between two documents

    Parameters
    ----------
    x: 1-d signature array of document with docid=x (int)
    a: 1-d signature array of document with docid=a (int)
    signature_matrix: stores signature vectors of all documents as columns (pandas.DataFrame)

    Returns
    -------
    the value of Jaccard similarity between documents x and a
    """
    x = signature_matrix[x]
    a = signature_matrix[a]
    return sum(x & a)/sum(x | a)


def euclid(x, a, signature_matrix):
    """Calculates the euclidean similarity between two documents

    Parameters
    ----------
    x: 1-d signature array of document with docid=x (int)
    a: 1-d signature array of document with docid=a (int)
    signature_matrix: stores signature vectors of all documents as columns (pandas.DataFrame)

    Returns
    -------
    the value of euclidean distance between documents x and a
    """
    x = signature_matrix[x]
    a = signature_matrix[a]
    return np.sum(a**2 - x**2)**0.5


def cosine(x, a, signature_matrix):
    """Calculates the cosine similarity between two documents

    Parameters
    ----------
    x: 1-d signature array of document with docid=x (int)
    a: 1-d signature array of document with docid=a (int)
    signature_matrix: stores signature vectors of all documents as columns (pandas.DataFrame)

    Returns
    -------
    the value ofcosine similarity between documents x and a
    """
    x = signature_matrix[x]
    a = signature_matrix[a]
    return np.dot(a, x)/(np.sum(a**2) * np.sum(x**2))**0.5


def compute_similarity(x, similar_documents, signature_matrix, sim_type="jaccard"):
    """Calculates the  cosine similarity between two documents

    Parameters
    ----------
    x: 1-d signature array of document with docid=x (int)
    similar_documents: a list of docids which are similar to x.
    signature_matrix: stores signature vectors of all documents as columns (pandas.DataFrame)
    sim_type: Possible values jaccard, euclid, cosine (Default = jaccard) 

    Returns
    -------
    list : sorted list of (docid, score) tuples.
    """
    if sim_type == "jaccard":
        sim_fun = jaccard
    elif sim_type == "euclid":
        sim_fun = euclid
    elif sim_type == "cosine":
        sim_fun = cosine
    # write for all other funcs
    ranked_list = []
    for i in similar_documents:
        if i == x:
            continue
        score = sim_fun(x, i, signature_matrix)
        ranked_list.append((i, score))

    if sim_type == "euclid":
        return sorted(ranked_list, key=lambda x: x[1], reverse=False)
    else:
        return sorted(ranked_list, key=lambda x: x[1], reverse=True)


def precision(threshold, output):
    """Calculates the cosine similarity between two documents

    Parameters
    ----------
    threshold: value of similarity above which retrieved docs are considered relevant (float)
    output:  list of retrieved items.

    Returns
    -------
    precision value for the given set of retrieved items.(float)
    """
    req = [i for f, i in output if i >= threshold]
    return len(req)/len(output)


def recall(threshold, x, size, output, signature_matrix, sim_type):
    """Calculates the  cosine similarity between two documents

    Parameters
    ----------
    threshold: value of similarity above which retrieved docs are considered relevant (float)
    x: 1-d signature array of document with docid=x (int)
    size: number of all documents in the corpus (int)
    output: list of retrieved items.
    signature_matrix: stores signature vectors of all documents as columns (pandas.DataFrame)
    sim_type: Possible values jaccard, euclid, cosine (str) 

    Returns
    -------
    returns recall value for the given set of retrieved items.
    """
    docs = compute_similarity(
        x, [i for i in range(size)], signature_matrix, sim_type)
    req = [i for f, i in output if i >= threshold]
    den = [i for f, i in docs if i >= threshold and f != x]
    if len(den) == 0:
        return "not defined"
    return len(req)/len(den)


def get_name_of_file(file_path, files):
    """Calculates the  cosine similarity between two documents

    Parameters
    ----------
    threshold: value of similarity above which retrieved docs are considered relevant(float)
    files: list of tuples containing filename and file id.

    Returns
    -------
    returns name of the file with given file_path.
    """
    for filename, f_id in files:
        if file_path == f_id:
            return filename
