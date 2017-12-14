#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import Counter
from unicodedata import normalize
import json
import re
import csv
import editdistance
import numpy as np
from multiprocessing import Process
import multiprocessing
from sklearn.cluster import DBSCAN

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

v=9998
MATRIX = np.full((v,v), np.inf)

def fillMatrix(subDataSet,i):
	print ("inicio do processo: " +str(i)) 
	for v1 in [v]:#[100,200,400,500,1000,9000]:
		
		with open("vagas/Dataset-Treino-Anonimizado-3.json") as data_file:
			dataSet = json.load(data_file)

		grafo = {}
		setVisitados = set([])
		for vaga1 in subDataSet:
			textoVaga1 = toClean(vaga1['title']+" "+vaga1['description'])
			#vagaTexto[int(vaga1['id'])] = textoVaga1
			grafo[int(vaga1['id'])] = set()


			print ("DBSCAN "+vaga1['id']+"\n")
			for vaga2 in dataSet:
				if (MATRIX[int(vaga1['id'])][int(vaga2['id'])] ==np.inf):
					if vaga1['id']!= vaga2['id']:
						textoVaga2 = toClean(vaga2['title']+" "+vaga2['description'])
						#vagaTexto[int(vaga2['id'])] = textoVaga2
						#d = levenshtein(textoVaga1,textoVaga2)
						d = int(editdistance.eval(textoVaga1,textoVaga2))
						valor = d/float(max( len(textoVaga2),len(textoVaga1)))
						MATRIX[int(vaga1['id'])][int(vaga2['id'])] = valor
						MATRIX[int(vaga2['id'])][int(vaga1['id'])] = valor
					else:
						MATRIX[int(vaga1['id'])][int(vaga2['id'])] = 0
						MATRIX[int(vaga2['id'])][int(vaga1['id'])] = 0

	print("fim do processo:" +str(i))

if __name__ == '__main__':
	with open("vagas/Dataset-Treino-Anonimizado-3.json") as data_file:
		docs = json.load(data_file)

	print (docs[9997])

	# manager = multiprocessing.Manager()
	# resultMultiP = manager.dict()
	# numberMultiP = 4
	# procs = []
	# aux = 0
	# tam = len(docs)
	# for numberP in range(1,numberMultiP+1):
	# 	print (aux,int((numberP*tam)/numberMultiP))
	# 	docsAux = docs[aux:int((numberP*tam)/numberMultiP)]
	# 	proc = Process(target=fillMatrix, args=(docsAux,numberP))
	# 	procs.append(proc)
	# 	proc.start()

	# 	aux = int((numberP*tam)/numberMultiP)
	# for proc in procs:
	# 	proc.join()
	# print ("acabou todas")

	fillMatrix(docs, 1)
	estimator = DBSCAN(eps=0.3, min_samples=1,metric='precomputed')
	estimator.fit(MATRIX)
	labels = estimator.labels_
	labels = [x+1 for x in labels]
	print (labels)

	with open("gabaritos/"+"DBSCAN"+str(v)+"vagas.csv", "w") as csv_file:
		writer = csv.writer(csv_file, delimiter=',')
		for vaga in docs:
			ID = int(vaga['id'])
			writer.writerow([ID, labels[ID]])

					
