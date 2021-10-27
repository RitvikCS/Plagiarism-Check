"""
The project aims for plagiarism detection using Locality Sensitive Hahsing(LSH)

Process:
    - Shingling :- Changing documents into set of tokens.Any substring of length 
    of k can be called a k-shingle.Here we generate a sparse incidence matrix.
    - Minhashing :- In the last step we got incidence matrix which is sparse so 
    here we target to reduce the size of matrix by producing signature for each
    document present in the incidence matrix
    - LSH :- Here we divide the signature matrix into horizontal bands and applying hash
    functions.We use a hash function  for hashing every band and while hashing a band,its 
    parts of each document are hashed into a set of buckets.So documents 
    with the same signature  in the band will land into the same buckets

"""

import time
import os
import shingling
import minhashing
import lsh
import statistics


def startLSH():
    print("\n*** --- LSH Plagiarism detector --- ***\n")

    # Shingling
    timer_start = time.time()
    shingle_size = 4
    folderpath = "corpus"
    extension = ".txt"
    shingle_matrix, files = shingling.get_shingle_matrix(
        folderpath, shingle_size, extension)
    print(shingle_matrix.shape)
    print(f"Time for Shingling: {time.time()-timer_start}")

    # Minhashing
    start_time = time.time()
    no_of_hash_functions = 50   # number of hash functions for signature matrix
    signature_matrix = minhashing.generate_signature_matrix(
        shingle_matrix, no_of_hash_functions)
    print(f"Time for minhashing: {time.time()-start_time}")

    # LSH
    start_time = time.time()
    r = 2
    buckets_list = lsh.generate_bucket_list(signature_matrix, r)
    print(f"Time for LSH: {time.time()-start_time}")

    sim_type = "jaccard"
    while True:
        testing_filepath = input("Enter path of file: ")
        if testing_filepath == "DONE":
            break
        if not os.path.exists(testing_filepath):
            print(">> The path doesnt exist.")
            continue
        for name, num in files:
            if testing_filepath == name:
                testing_filepath = int(num)

        threshold = float(input("Enter threshold: "))

        print(f"Input Given file: {statistics.get_name_of_file(testing_filepath, files)}")
        similar_documents = lsh.find_similar_documents(
            testing_filepath, buckets_list, signature_matrix, r)
        output = statistics.compute_similarity(
            testing_filepath, similar_documents, shingle_matrix, sim_type)

        for file_path, score in output:
            if score > threshold:
                print(f"{statistics.get_name_of_file(file_path, files)}\t{score}")

        print(f"Precision: {statistics.precision(threshold, output)}")
        print(f"Recall: {statistics.recall(threshold, testing_filepath, len(files), output, shingle_matrix, sim_type)}")


if __name__ == "__main__":
    startLSH()
