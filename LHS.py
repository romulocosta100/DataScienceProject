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

M = np.empty([1,1])


def remover_acentos(txt, codif='utf-8'):
	return normalize('NFKD', txt).encode('ASCII','ignore')

def toClean(text):
	#remove acentos
	text = remover_acentos(text)

	text = re.sub('[:|;|/]',' ',text)
	#remove caracteres especiais
	text = re.sub('[@|%|?|$|!|#|.|,|(|)|&|+|-|-|=|*]','',text)
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

	print ('app funhash: %f'  %timeFun)
	print ('fill matrix: %f'   %timeSh)
	return matrix


def dfs_paths(graph):
	aux = 0
	
	itemGrupo = {}

	visited = set()
	for node in graph.keys():
		if node not in visited:
			aux+=1
			start = node
			print "[ ",
			stack = [start]	
			while stack:
				vertex = stack.pop()
				if vertex not in visited:
					itemGrupo[vertex] = aux
					print vertex,
					visited.add(vertex)
					for neighbor in graph[vertex]:
						stack.append(neighbor)
			print " ]\n"

	return itemGrupo

def concatList(A):
	out = ""
	for a in A:
		out+=str(a)
	return out

def localitySensitiveHashing(r, band, matrix,t):
	start = time.time()
	row, col = np.shape(matrix)
	inicio = 0
	
	bandBuckets = [{} for _ in xrange(band)]

	hashFunc = randomHash(100)

	for b in xrange(band):
		signaturePerBands = np.apply_along_axis(concatList,0, matrix[inicio:inicio+r,:])

		
		for i in xrange(col):
			hashValue = hashFunc(signaturePerBands[i])

			if hashValue in bandBuckets[b]:
				bandBuckets[b][hashValue].add(i)
			else:
				bandBuckets[b][hashValue] = set([i])


		inicio = inicio+r
	end = time.time()
	print ('Calcular buckets : %f' % (end - start))

	grafo = {}

	start = time.time()
	for buckets in bandBuckets:

		for key,bucket in buckets.items():

			for a in bucket:
				for b in bucket:
					if(a!=b):
						#Memorizacao
						if(M[a][b]==np.inf):
							M[a][b] = compute_jaccard_index(set(matrix[:,a]),set(matrix[:,b]))
							M[b][a] = M[a][b]

						if(M[a][b]>=t):
							if(a in grafo):
								grafo[a].add(b)
							else:
								grafo[a] = set([b])

	dfs_paths(grafo)
	end = time.time()
	print ('Calcular distancias : %f' % (end - start))




if __name__ == "__main__":

	#for v in [200,400,800,1600,3200]:
	#print str(v)," vagas :"
	v = 200
	with open(str(v)+'vagas.json') as data_file:
		dataSet = json.load(data_file)

	docs = []
	k = 6

	start = time.time()
	for vaga1 in dataSet:
		textoVaga1 = toClean(vaga1['title'] + " " + vaga1['description'])

		docs.append(hashed_k_shingle(textoVaga1,6))
	end = time.time()
	print ('limpar texto: %f' % (end - start))


	start = time.time()
	matrix = minhashSignature(docs,200)
	end = time.time()
	print ('Construir a matrix : %f' % (end - start))

	
	global M
	M = np.full((v,v), np.inf)
	localitySensitiveHashing(20,10,matrix,0.8)


	print "\n*************************\n"

	