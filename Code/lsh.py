"""
Hashing similar documents to same buckets to know the similar documents.
"""


def band_hashing(band, hash_f, buckets_dictionary):
    """Helper-method: performs hash on bands.Input is band and it hashes it
     and puts in buckets_list at its respective postition.
    """

    for col in band.columns:
        h = hash_f(tuple(band[col].values))
        if h in buckets_dictionary:
            buckets_dictionary[h].append(col)
        else:
            buckets_dictionary[h] = [col]


def generate_bucket_list(signature_matrix, r, hash_f=None):
    """
    This function generates and returns the list of dictionaries where 
    each band is hashed to a bucket in a dictionary.

    Parameters
    ----------
    signature_matrix: signatures of all the documents generated from minhashing (DataFrame)
    r: number of rows in each band (int)
    hash_f: hash function for hashing documents into buckets (optional)

    Returns
    -------
    buckets_list: Each dictionary contains hashes of column vectors of the
    band as keys and the list of documents as values.
    """
    # b: number of bands
    # r: number of rows in a band
    # n: document signature length

    n = signature_matrix.shape[0]
    b = n//r
    buckets_list = [dict() for i in range(b)]

    if hash_f == None:
        hash_f = hash

    for i in range(0, n-r+1, r):
        band = signature_matrix.loc[i:i+r-1, :]
        band_hashing(band, hash_f, buckets_list[int(i/r)])

    return buckets_list


def query_band_hashing(band, hash_f):
    """helper-function: To Perform hash on query doc bands
    """

    hash_list = []
    h = hash_f(tuple(band.values))
    hash_list.append(h)

    return hash_list


def find_similar_documents(doc_id, buckets_list, signature_matrix, r, hash_f=None):
    """Finds the  similar documents

    Parameters
    ----------
    buckets_list: list of dictionary objects generated by generate_bucket_list
    hash_f: the same hash function used for generate_bucket_list (optional)

    Returns
    -------
    Returns a set containing similar documents to given document
    """

    n = signature_matrix.shape[0]
    b = n//r

    if hash_f == None:
        hash_f = hash

    query_bucket_list = []

    for i in range(0, n-r+1, r):
        band = signature_matrix.loc[i:i+r-1, int(doc_id)]
        query_bucket_list.append(query_band_hashing(band, hash_f))

    similar_documents = set()
    for i in range(len(query_bucket_list)):
        for j in range(len(query_bucket_list[i])):
            similar_documents.update(
                set(buckets_list[i][query_bucket_list[i][j]]))

    return similar_documents


if __name__ == '__main__':
    from minhashing import minhash
    from shingling import main
    data = main()
    signature_matrix = minhash(data)
