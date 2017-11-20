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


if __name__ == "__main__":
	#sentence = 'this is a foo bar sentences and i want to ngramize it'
	#shingles = hashed_k_shingle(sentence,6)
	#print (shingles)

	# doc1 = set([0,3])
	# doc2 = set([2])
	# doc3 = set([1,3,4])
	# doc4 = set([0,2,3])
    #
	# docs = [doc1,doc2,doc3,doc4 ]
    #
	# print minhashSignature(docs,10)


	with open('15vagas.json') as data_file:
		dataSet = json.load(data_file)

	docs = []
	k = 6
	for vaga1 in dataSet:
		textoVaga1 = toClean(vaga1['title'] + " " + vaga1['description'])

		docs.append(hashed_k_shingle(textoVaga1,6))


	matrix = minhashSignature(docs,95)

	textoVaga1 = toClean(dataSet[1]['title'] + " " + dataSet[1]['description'])
	textoVaga8 = toClean(dataSet[4]['title'] + " " + dataSet[4]['description'])
	print  textoVaga1
	print "________________________"
	print  textoVaga8
	print "real:     ", jaccardSimilarity(hashed_k_shingle(textoVaga1,6) , hashed_k_shingle(textoVaga8,6))

	col1 = matrix[:,1]
	col8 = matrix[:,4]
	print "estimado: ",jaccardSimilarity(set(col1), set(col8) )