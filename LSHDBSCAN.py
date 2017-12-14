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
import random
import time
import timeit
from multiprocessing import Process
import multiprocessing
from sklearn.cluster import DBSCAN


M = np.empty([1,1])
grafo = {}

def remover_acentos(txt, codif='utf-8'):
	return normalize('NFKD', txt).encode('ASCII','ignore')

def toClean(text):
	#remove acentos
	text = remover_acentos(text)

	text = re.sub('[:|;|/]',' ',text.decode("utf-8"))
	#remove caracteres especiais
	text = re.sub('[@|%|?|$|!|#|.|,|(|)|&|+|-|-|=|*]','',text )
	#remove multiplos \n's #remove multiplos espacos
	text = " ".join(text.split())
	#todos caracteres em lowercase
	text = text.lower()

	return text

def compute_jaccard_index(set_1, set_2):
	n = len(set_1.intersection(set_2))
	return n / float(len(set_1) + len(set_2) - n) 
	
def jaccardSimilarity(S,T):
	return len( S & T )/float(len ( S | T ))

# "hasheia" o k-shingle de um doc
def hashed_k_shingle(text,k):
	seed=42

	shingles = set()
	kgrams = ngrams(list(text), k)
	for grams in kgrams:
		shingles.add(  mmh3.hash( "".join(map(str,grams)),  seed  ) )
		

	return shingles

def hashed_k_shingle_numpy(text,k):
	seed=42

	shingles = np.array([])
	#shingles = set()
	kgrams = ngrams(list(text), k)
	for grams in kgrams:
		#shingles.add(  mmh3.hash( "".join(map(str,grams)),  seed  ) )
		shingles = np.append(shingles,mmh3.hash( "".join(map(str,grams)),  seed  ))

	return np.unique(shingles)


def randomHash(seed):
	def f(x):
		return mmh3.hash(str(x),seed)
	return f


def fooHash(a):
	def f(x):
		return (a*x + 1) % 5
	return f


#constrói a assinatura minHashing de cada doc
def minhashSignature(docs,nhash):
	matrix =  np.full((nhash, len(docs)), np.inf)
	randomSeeds = range(nhash)
	hashs = [randomHash(seed) for seed in randomSeeds ]

	col = 0
	timeFun = 0
	timeSh  = 0
	for doc in docs:
		start = time.time()
		for shingle in doc:
			start = time.time()
			shingleHashs = [ funHash(shingle) for funHash in hashs ]
			end = time.time()
			timeFun += (end-start)
			#print ('HS     : %f' % (end - start))

			start = time.time()
			# row = 0
			# for sh in shingleHashs:
			# 	matrix[row][col] = min(matrix[row][col],sh)
			# 	row+=1
			matrix[:,col] = np.minimum(matrix[:,col],shingleHashs)
			end = time.time()
			timeSh += (end-start)
			#print ('min sh : %f' % (end - start))
		end = time.time()
		#print ('make line: %f' % (end - start))

		col+=1

	# print ('app funhash: %f'  %timeFun)
	# print ('fill matrix: %f'   %timeSh)
	#print (resultMultiP[numberP])
	
	return matrix

#constrói a assinatura minHashing de cada doc
def minhashSignatureMultiP(numberP,docs,nhash,resultMultiP):
	print("inicio processo "+str(numberP))
	matrix =  np.full((nhash, len(docs)), np.inf)
	randomSeeds = range(nhash)
	hashs = [randomHash(seed) for seed in randomSeeds ]

	col = 0
	timeFun = 0
	timeSh  = 0
	for doc in docs:
		start = time.time()
		for shingle in doc:
			start = time.time()
			shingleHashs = [ funHash(shingle) for funHash in hashs ]
			end = time.time()
			timeFun += (end-start)
			#print ('HS     : %f' % (end - start))

			start = time.time()
			# row = 0
			# for sh in shingleHashs:
			# 	matrix[row][col] = min(matrix[row][col],sh)
			# 	row+=1
			matrix[:,col] = np.minimum(matrix[:,col],shingleHashs)
			end = time.time()
			timeSh += (end-start)
			#print ('min sh : %f' % (end - start))
		end = time.time()
		#print ('make line: %f' % (end - start))

		col+=1

	# print ('app funhash: %f'  %timeFun)
	# print ('fill matrix: %f'   %timeSh)
	resultMultiP[numberP] = matrix
	#print (resultMultiP[numberP])
	print("fim processo "+str(numberP))
	#return matrix

#constrói a assinatura minHashing de cada doc
def minhashSignature2(docs,nhash):
	matrix =  np.full((nhash, len(docs)), np.inf)
	randomSeeds = range(nhash)
	hashs = [ randomHash(seed) for seed in randomSeeds ]

	col = 0
	row = 0
	timeFun = 0
	timeSh  = 0

	for doc in docs:
		#doc = frozenset(doc)
		lenDoc = len(doc)#doc.size
		colaux = 0
		matrixShingleFuncs = np.full((len(doc), nhash),np.inf)
		for funhash in hashs:
			funhash = np.vectorize(funhash)
			matrixShingleFuncs[:,colaux] = funhash(doc)
			end = time.time()
			colaux+=1

		for i in range(lenDoc):
			matrix[:,col] = np.minimum(matrix[:,col],matrixShingleFuncs[i,:])

		#matrix[:,col] = np.apply_along_axis(min,0, matrixShingleFuncs)


		col+=1		

	# print ('app funhash: %f'  %timeFun)
	# print ('fill matrix: %f'   %timeSh)
	return matrix


def dfs_paths(graph):
	aux = 0
	
	itemGrupo = {}

	visited = set()
	for node in graph.keys():
		if node not in visited:
			aux+=1
			start = node
			#print ("[ ",)
			stack = [start]	
			while stack:
				vertex = stack.pop()
				if vertex not in visited:
					itemGrupo[vertex] = aux
					#print (str(vertex)+" ")
					visited.add(vertex)
					for neighbor in graph[vertex]:
						stack.append(neighbor)
			#print (" ]\n")

	return itemGrupo

def concatList(A):
	out = ""
	for a in A:
		out+=str(a)
	return out

def localitySensitiveHashing(r, band, matrix,nhash,t,docsshingles,docs,fileName,idsDocs):
	start = time.time()
	row, col = np.shape(matrix)
	inicio = 0
	
	bandBuckets = [{} for _ in range(band)]

	hashFunc = randomHash(nhash)

	for b in range(band):
		signaturePerBands = np.apply_along_axis(concatList,0, matrix[inicio:inicio+r,:])
		for i in range(col):
			hashValue = hashFunc(signaturePerBands[i])

			if hashValue in bandBuckets[b]:
				bandBuckets[b][hashValue].add(i)
			else:
				bandBuckets[b][hashValue] = set([i])


		inicio = inicio+r
	end = time.time()
	print ('Calcular buckets : %f' % (end - start))

	
	cont = 0
	for buckets in bandBuckets:

		for key,bucket in buckets.items():

			for a in bucket:
				for b in bucket:
					if(a!=b):
						#Memorizacao
						if(M[a][b]==2):
							# M[a][b] = compute_jaccard_index(set(matrix[:,a]),set(matrix[:,b]))
							M[a][b] = editdistance.eval(docs[a], docs[b])/float(max( len(docs[a]),len(docs[b])))
							M[b][a] = M[a][b]
				else:
					cont += 1
	start = time.time()
	estimator = DBSCAN(eps=1-t, min_samples=1,metric='precomputed')
	estimator.fit(M)
	labels = estimator.labels_
	labels = [x+1 for x in labels]
	#print (labels)
	end = time.time()
	print ('DBSCAN : %f' % (end - start))
		
	with open("resultados/resultado_final.csv", "w") as csv_file:
			writer = csv.writer(csv_file, delimiter=',')
			for ID in idsDocs:
				writer.writerow([ID, labels[ID]])

def run(nvagas,nhash,rows,bands,t,k):
	with open("vagas/COnjuntoTeste.json") as data_file:
		dataSet = json.load(data_file)

	idsDocs = []
	docs = []
	docstext = []
	nvagas = len(dataSet)
	print ("nvagas:", nvagas)
	start = time.time()
	for vaga1 in dataSet:
		idsDocs.append(int(vaga1['id']) )
		textoVaga1 = toClean(vaga1['title'] + " " + vaga1['description'])
		docstext.append(textoVaga1)
		docs.append(hashed_k_shingle(textoVaga1,k))
		grafo[int(vaga1['id'])] = set()
	end = time.time()
	print ('limpar texto: %f' % (end - start))

	#*********** criar matriz multiprocessing********************
	manager = multiprocessing.Manager()
	resultMultiP = manager.dict()
	numberMultiP = 4
	procs = []
	aux = 0
	tam = len(docs)
	for numberP in range(1,numberMultiP+1):
		#print (aux,int((numberP*tam)/numberMultiP))
		docsAux = docs[aux:int((numberP*tam)/numberMultiP)]
		proc = Process(target=minhashSignatureMultiP, args=(numberP,docsAux,nhash,resultMultiP))
		procs.append(proc)
		proc.start()

		aux = int((numberP*tam)/numberMultiP)
	

	for proc in procs:
		proc.join()

	print("juntar matrix")
	keys = sorted(resultMultiP.keys())
	matrix = resultMultiP[keys[0]]
	keys.remove(keys[0])
	
	for key in keys:
		matrix = np.concatenate((matrix, resultMultiP[key]), axis=1)

	#print (matrix)
	#*********** criar matriz multiprocessing********************
	fileName = "DBSCANvagas_"+str(nvagas)+"_hash_"+str(nhash)+"_rows_"+str(rows)+"_bands_"+str(bands)+"_k_"+str(k)+".csv"

	global M
	M = np.full((nvagas,nvagas), 2)
	np.fill_diagonal(M, 0)
	localitySensitiveHashing(rows,bands,matrix,nhash,t,docs,docstext,fileName,idsDocs)

if __name__ == "__main__":
	nvagas = 48773
	nhash = 200
	rows = 10
	bands = 20
	t = 0.8
	k = 7

	for nhash in [200]:
		for rows in [10]:
			print ('nhash: ', nhash, ' row:', rows)
			bands = int(nhash/rows)
			start = time.time()
			run(nvagas,nhash,rows,bands,t,k)
			end = time.time()
			print ('execução: %f' % (end-start))

	# for k in range(3,20):
	# 	start = time.time()
	# 	run(nvagas,nhash,rows,bands,t,k)
	# 	end = time.time()
	# 	print(k)
	# 	print ('execução : %f' % (end - start))



