"""
Generate the incidence matrix for corpus by using Shingling
"""

from pandas import DataFrame, read_pickle
import numpy as np
import codecs
import os
import pickle
from tqdm import tqdm


def list_all_files(folderpath, extension=".txt"):
    """
    Build the corpus files list from the folder path given.

    Parameters
    ----------
    folderpath: The path to corpus files(str)
    extension:  The extension of the files to be read. (Default: .txt)

    Returns
    -------
    list : a list of files present in the directory(includes sub-folders)

    Raises
    ------
    Exception
        if given folder path does not exist
    """

    print(f"Reading given corpus: {folderpath}")
    # check if folder path exists
    if(not os.path.exists(folderpath)):
        raise Exception(f"{folderpath} does not exist")
        return

    document_files = []
    i = 0           # index/id to identify each file

    if extension == None:
        extension = ""

    for (root, dirs, files) in os.walk(folderpath):
        for f in files:
            if f.endswith(extension):
                document_files.append((os.path.join(root, f), i))
                i += 1

    return document_files


def build_matrix(files, k=4, newline=False):
    """helper: build incidence matrix for k-grams (shingles)
    """

    df = DataFrame(columns=[x[1] for x in files])

    for f in tqdm(files):
        with codecs.open(f[0], 'r', encoding="utf8", errors='ignore') as doc:
            # print("reading: "+f[0])
            data = doc.read()
            # df[ f[1] ] = 0
            data = data.lower()             # lowercase all letters
            # substiture multiples spaces with single space
            data = ' '.join(data.split())
            # replace windows line endings with space
            data = data.replace('\r\n', ' ')
            data = data.replace('\r', '')   # remove \r in windows
            data = data.replace('\t', '')   # remove tab-spaces
            if newline is False:
                data = data.replace('\n', ' ')

            # st_time = time.time()
            for i in range(0, len(data)-k+1):
                shingle = data[i:i+k]
                if (shingle in df.index) == False:
                    df.loc[shingle] = [0 for i in range(df.shape[1])]
                df.at[shingle, f[1]] = 1
            # print(time.time()-st_time)

    return df


def get_shingle_matrix(folderpath, shingle_size=8, extension=".txt"):
    """Does Shingling and builds incidence index for shingles. Please note that if 
    a already generated pickle is present in the folder then this function automatically loads

    Parameters
    ----------
    folderpath: The path to corpus files (str)
    shingle_size: Size of shingles for dividing the documents.(Default is  8)
    extension: The extension of the files to be read. (Default: .txt)

    Returns
    -------
    pandas.Dataframe : dataframe containing rows as shingles and cols as doc_ids
    """

    inci_mat = None
    if os.path.exists(f"{folderpath}_inc_mat.pickle"):
        inci_mat = read_pickle(f"{folderpath}_inc_mat.pickle")
        if os.path.exists("file_list.pickle"):
            print(f"Using previously created {folderpath}_inc_mat.pickle file")
            print("Using pickled file list")
            with open("file_list.pickle", 'rb') as file_list_pkl:
                files = pickle.load(file_list_pkl)
            return inci_mat, files
        print("file list not found")

    files = list_all_files(folderpath, extension)

    inci_mat = build_matrix(files, k=shingle_size)

    print("Saving generated incidence index to file...")
    inci_mat.to_pickle(f"{folderpath}_inc_mat.pickle")
    with open("file_list.pickle", 'wb') as file_list_pkl:
        pickle.dump(files, file_list_pkl)
    print(f"Saved to {folderpath}_inc_mat.pickle")
    return inci_mat, files


def main():
    from time import time
    st_time = time()
    folderpath = "corpus"
    inci_mat = get_shingle_matrix(folderpath)
    print(f"endtime: {time() - st_time}")
    print(inci_mat)


if __name__ == "__main__":
    main()
