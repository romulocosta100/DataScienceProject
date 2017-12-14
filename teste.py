# coding: utf-8
import pandas as pd
import numpy as np
import json
import csv
import editdistance
import re
from nltk import ngrams
from unicodedata import normalize
from numpy import array

def buildMatrix(path):
	datapath=path
	gb_fb = pd.read_csv(datapath)
	a=gb_fb.groupby('1')['0'].apply(list)
	lil=a.tolist()
	# print(lil)
	lilnp=np.array(lil)
	# print(lilnp)
	return lilnp

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

def calcPrecisionRecall(gab, resp):
	M = [0,0,0]
	#quantos estao em gab e estao/nao estao em resp
	for cluster in gab:
		# print "c--", cluster, type(cluster)
		if len(cluster) == 1:
			for r in resp:
				if len(cluster) == 1:
					if cluster[0] == r[0]:
						M[0]+=1
		else:
			for i in xrange(len(cluster)):
				for j in xrange(i+1,len(cluster)):
					a = set([cluster[i], cluster[j]])
					flag = True
					for r in resp:
						if flag:	
							for k in xrange(len(r)):
								for l in xrange(k+1,len(r)):
									b = set([r[k], r[l]])
									# print a, "--",b
									if a == b:
										M[0]+=1
										flag = False
										break
					if flag:
						M[1] += 1

	#quantos estao em resp e nao estao
	# print "********************************"
	for cluster in resp:
	# 	print "c--", cluster, type(cluster)
		for i in xrange(len(cluster)):
			for j in xrange(i+1,len(cluster)):
				a = set([cluster[i], cluster[j]])
				flag = True
				for r in gab:
					if flag:	
						for k in xrange(len(r)):
							for l in xrange(k+1,len(r)):
								b = set([r[k], r[l]])
								# print a, "--",b
								if a == b:
									flag = False
									break
				if flag:
					M[2] += 1					
	recall = M[0]/ (1.0*(M[0]+M[1]))
	precision = M[0] / (1.0*(M[0]+M[2]))
	print "precision: ", precision
	print "recall: ", recall
	print "f-measure: ", 2* (precision*recall)/ (precision+recall)
	return M

if __name__ == "__main__":
	nvagas = 9998
	nhash = 300
	rows = 10
	bands = 30
	t = 0.8
	k = 7
	# for k in range(3,19	):
	respath = "resultados/nhash/DBSCANvagas_"+str(nvagas)+"_hash_"+str(nhash)+"_rows_"+str(rows)+"_bands_"+str(bands)+"_k_"+str(k)+".csv"
	gabpath = "gabaritos/DBSCAN9998vagas.csv"
	mg = buildMatrix(gabpath)
	mr = buildMatrix(respath)
	# print (str(k)+"-shingle:")
	m = calcPrecisionRecall(mg, mr)

	print (m, "************\n")

	# with open('vagas/Dataset-Treino-Anonimizado-3.json') as data_file:
	# 	dataSet = json.load(data_file)

	# a = 98
	# b = 5491
	# c = 7672
	# for i in [3283,3285,4238,4279,4290,4470,4990,4991,5043,5660,5661,6070,7234,7455,7806]:
	# 	vaga1 = dataSet[i]
	# 	print (toClean(vaga1['title']+" "+vaga1['description']))
	# 	print ("**********************************************")

	# vaga2 = dataSet[b]
	# vaga3 = dataSet[c]
	# print (toClean(vaga1['title']+" "+vaga1['description']))
	# print ("**********************************************")
	# print (toClean(vaga2['title']+" "+vaga2['description']))
	# print ("**********************************************")
	# print (toClean(vaga3['title']+" "+vaga3['description']))
	# with open('resultados/DBSCAN100vagas.csv', 'r') as csvfile:
	# 	reader = csv.reader(csvfile, delimiter=',')
	# 	listFile = list(reader)
	# 	for i in xrange(len(listFile)):
	# 		for j in xrange(i+1,len(listFile)):
	# 			if listFile[i][1] == listFile[j][1]:
	# 				a = listFile[i][0]
	# 				b = listFile[j][0]

	# 				textoVaga1 = toClean(dataSet[int(a)]['title']+" "+dataSet[int(a)]['description'])
	# 				textoVaga2 = toClean(dataSet[int(b)]['title']+" "+dataSet[int(b)]['description'])
	# 				d = int(editdistance.eval(textoVaga1,textoVaga2))
	# 				valor = 1 - d/float(max( len(textoVaga2),len(textoVaga1)) )
	# 				if valor < 0.8:
	# 					print "O GABARITO TÁ ERRADO!"
	# 					print b, a, ">>", valor
	# 					break