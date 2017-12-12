#!/usr/bin/env python
# -*- coding: utf-8 -*-
from unicodedata import normalize
import json
import numpy as np
import re
import csv
import mmh3
import editdistance
from nltk import ngrams
import time
from multiprocessing import Process
import multiprocessing
from sklearn.cluster import DBSCAN

M = np.empty([1, 1])
grafo = {}


def remover_acentos(txt, codif='utf-8'):
    return normalize('NFKD', txt).encode('ASCII', 'ignore')


def to_clean(text):
    # remove acentos
    text = remover_acentos(text)

    text = re.sub('[:|;|/]', ' ', text.decode("utf-8"))
    # remove caracteres especiais
    text = re.sub('[@|%|?|$|!|#|.|,|(|)|&|+|-|-|=|*]', '', text)
    # remove multiplos \n's #remove multiplos espacos
    text = " ".join(text.split())
    # todos caracteres em lowercase
    text = text.lower()

    return text


def compute_jaccard_index(set_1, set_2):
    n = len(set_1.intersection(set_2))
    return n / float(len(set_1) + len(set_2) - n)


def jaccard_similarity(S, T):
    return len(S & T) / float(len(S | T))


# "hasheia" o k-shingle de um doc
def hashed_k_shingle(text, k):
    seed = 42

    shingles = set()
    kgrams = ngrams(list(text), k)
    for grams in kgrams:
        shingles.add(mmh3.hash("".join(map(str, grams)), seed))

    return shingles


def hashed_k_shingle_numpy(text, k):
    seed = 42

    shingles = np.array([])
    # shingles = set()
    kgrams = ngrams(list(text), k)
    for grams in kgrams:
        # shingles.add(  mmh3.hash( "".join(map(str,grams)),  seed  ) )
        shingles = np.append(shingles, mmh3.hash("".join(map(str, grams)), seed))

    return np.unique(shingles)


def random_hash(seed):
    def f(x):
        return mmh3.hash(str(x), seed)

    return f


def foo_hash(a):
    def f(x):
        return (a * x + 1) % 5

    return f


# constrói a assinatura minHashing de cada doc
def minhash_signature(docs, nhash):
    matrix = np.full((nhash, len(docs)), np.inf)
    random_seeds = range(nhash)
    hashs = [random_hash(seed) for seed in random_seeds]

    col = 0
    time_fun = 0
    time_sh = 0
    for doc in docs:
        start = time.time()
        for shingle in doc:
            start = time.time()
            shingleHashs = [funHash(shingle) for funHash in hashs]
            end = time.time()
            time_fun += (end - start)
            # print ('HS     : %f' % (end - start))

            start = time.time()
            # row = 0
            # for sh in shingleHashs:
            # 	matrix[row][col] = min(matrix[row][col],sh)
            # 	row+=1
            matrix[:, col] = np.minimum(matrix[:, col], shingleHashs)
            end = time.time()
            time_sh += (end - start)
        # print ('min sh : %f' % (end - start))
        end = time.time()
        # print ('make line: %f' % (end - start))

        col += 1

    print('app funhash: %f' % time_fun)
    print('fill matrix: %f' % time_sh)
    # print (resultMultiP[numberP])

    return matrix


# constrói a assinatura minHashing de cada doc
def minhash_signature_multi_processing(numberP, docs, nhash, resultMultiP):
    print("inicio processo " + str(numberP))
    matrix = np.full((nhash, len(docs)), np.inf)
    random_seeds = range(nhash)
    hashs = [random_hash(seed) for seed in random_seeds]

    col = 0
    time_fun = 0
    time_sh = 0
    for doc in docs:
        start = time.time()
        for shingle in doc:
            start = time.time()
            shingle_hashs = [funHash(shingle) for funHash in hashs]
            end = time.time()
            time_fun += (end - start)
            # print ('HS     : %f' % (end - start))

            start = time.time()
            # row = 0
            # for sh in shingleHashs:
            # 	matrix[row][col] = min(matrix[row][col],sh)
            # 	row+=1
            matrix[:, col] = np.minimum(matrix[:, col], shingle_hashs)
            end = time.time()
            time_sh += (end - start)
        # print ('min sh : %f' % (end - start))
        end = time.time()
        # print ('make line: %f' % (end - start))

        col += 1

    print('app funhash: %f' % time_fun)
    print('fill matrix: %f' % time_sh)
    resultMultiP[numberP] = matrix
    # print (resultMultiP[numberP])
    print("fim processo " + str(numberP))


# return matrix


# constrói a assinatura minHashing de cada doc
def minhash_signature2(docs, nhash):
    matrix = np.full((nhash, len(docs)), np.inf)
    random_seeds = range(nhash)
    hashs = [random_hash(seed) for seed in random_seeds]

    col = 0

    for doc in docs:
        # doc = frozenset(doc)
        len_doc = len(doc)  # doc.size
        col_aux = 0
        matrix_shingle_funcs = np.full((len(doc), nhash), np.inf)
        for funhash in hashs:
            funhash = np.vectorize(funhash)
            matrix_shingle_funcs[:, col_aux] = funhash(doc)
            end = time.time()
            col_aux += 1

        for i in range(len_doc):
            matrix[:, col] = np.minimum(matrix[:, col], matrix_shingle_funcs[i, :])

        # matrix[:,col] = np.apply_along_axis(min,0, matrixShingleFuncs)

        col += 1

    # print ('app funhash: %f'  %timeFun)
    # print ('fill matrix: %f'   %timeSh)
    return matrix


def dfs_paths(graph):
    aux = 0

    item_grupo = {}

    visited = set()
    for node in graph.keys():
        if node not in visited:
            aux += 1
            start = node
            # print ("[ ",)
            stack = [start]
            while stack:
                vertex = stack.pop()
                if vertex not in visited:
                    item_grupo[vertex] = aux
                    # print (str(vertex)+" ")
                    visited.add(vertex)
                    for neighbor in graph[vertex]:
                        stack.append(neighbor)
                        # print (" ]\n")

    return item_grupo


def concat_list(A):
    out = ""
    for a in A:
        out += str(a)
    return out


def locality_sensitive_hashing(r, band, matrix, t, docs, file_name, ids_docs):
    start = time.time()
    row, col = np.shape(matrix)
    inicio = 0

    band_buckets = [{} for _ in range(band)]

    hash_func = random_hash(100)

    for b in range(band):
        signature_per_bands = np.apply_along_axis(concat_list, 0, matrix[inicio:inicio + r, :])

        for i in range(col):
            hash_value = hash_func(signature_per_bands[i])

            if hash_value in band_buckets[b]:
                band_buckets[b][hash_value].add(i)
            else:
                band_buckets[b][hash_value] = set([i])

        inicio = inicio + r
    end = time.time()
    print('Calcular buckets : %f' % (end - start))

    cont = 0
    for buckets in band_buckets:

        for key, bucket in buckets.items():

            for a in bucket:
                for b in bucket:
                    if a != b:
                        # Memorizacao
                        if M[a][b] == 2:
                            # M[a][b] = compute_jaccard_index(set(matrix[:,a]),set(matrix[:,b]))
                            M[a][b] = editdistance.eval(docs[a], docs[b]) / float(max(len(docs[a]), len(docs[b])))
                            M[b][a] = M[a][b]
                else:
                    cont += 1
    start = time.time()
    estimator = DBSCAN(eps=1 - t, min_samples=1, metric='precomputed')
    estimator.fit(M)
    labels = estimator.labels_
    labels = [x + 1 for x in labels]
    print(labels)
    end = time.time()
    print('DBSCAN : %f' % (end - start))

    with open("resultados/" + file_name, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for ID in ids_docs:
            writer.writerow([ID, labels[ID]])


def run(num_vagas, n_hash, rows, bands, t, k):
    with open("vagas/" + str(num_vagas) + "vagas.json") as data_file:
        data_set = json.load(data_file)

    ids_docs = []
    docs = []
    docstext = []
    start = time.time()
    for vaga1 in data_set:
        ids_docs.append(int(vaga1['id']))
        texto_vaga1 = to_clean(vaga1['title'] + " " + vaga1['description'])
        docstext.append(texto_vaga1)
        docs.append(hashed_k_shingle(texto_vaga1, k))
        grafo[int(vaga1['id'])] = set()
    end = time.time()
    print('limpar texto: %f' % (end - start))

    # *********** criar matriz multiprocessing********************
    manager = multiprocessing.Manager()
    result_multi_p = manager.dict()
    number_multi_p = 4
    procs = []
    aux = 0
    tam = len(docs)
    for numberP in range(1, number_multi_p + 1):
        # print (aux,int((numberP*tam)/numberMultiP))
        docsAux = docs[aux:int((numberP * tam) / number_multi_p)]
        proc = Process(target=minhash_signature_multi_processing, args=(numberP, docsAux, n_hash, result_multi_p))
        procs.append(proc)
        proc.start()

        aux = int((numberP * tam) / number_multi_p)

    for proc in procs:
        proc.join()

    print("juntar matrix")
    keys = sorted(result_multi_p.keys())
    matrix = result_multi_p[keys[0]]
    keys.remove(keys[0])

    for key in keys:
        matrix = np.concatenate((matrix, result_multi_p[key]), axis=1)

    # print (matrix)
    # *********** criar matriz multiprocessing********************
    file_name = "DBSCANvagas_" + str(num_vagas) + "_hash_" + str(n_hash) + "_rows_" + str(rows) + "_bands_" + str(
        bands) + ".csv"

    global M
    M = np.full((num_vagas, num_vagas), 2)
    np.fill_diagonal(M, 0)
    locality_sensitive_hashing(rows, bands, matrix, t, docstext, file_name, ids_docs)


if __name__ == "__main__":
    nvagas = 9000
    nhash = 200
    rows = 10
    bands = 20
    t = 0.8
    k = 6

    run(nvagas, nhash, rows, bands, t, k)