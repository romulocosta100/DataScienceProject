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
	for doc in docs:
		for shingle in doc:
			shingleHashs = [ funHash(shingle) for funHash in hashs ]

			row = 0
			for sh in shingleHashs:
				matrix[row][col] = min(matrix[row][col],sh)
				row+=1

		col+=1

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
	grafo = {}

	for buckets in bandBuckets:

		for key,bucket in buckets.items():

			for a in bucket:
				for b in bucket:
					if(a!=b):
						if(compute_jaccard_index(set(matrix[:,a]),set(matrix[:,b]))>=t):
							if(a in grafo):
								grafo[a].add(b)
							else:
								grafo[a] = set([b])

	dfs_paths(grafo)




if __name__ == "__main__":

	with open('Dataset-Treino-Anonimizado-3.json') as data_file:
		dataSet = json.load(data_file)

	docs = []
	k = 6

	for vaga1 in dataSet:
		textoVaga1 = toClean(vaga1['title'] + " " + vaga1['description'])

		docs.append(hashed_k_shingle(textoVaga1,6))

	matrix = minhashSignature(docs,200)

	print localitySensitiveHashing(20,10,matrix,0.8)
	