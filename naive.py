#!/usr/bin/env python
# -*- coding: utf-8 -*-
from unicodedata import normalize
import json
import numpy as np
import re
import csv

def levenshtein(source, target):
	if len(source) < len(target):
		return levenshtein(target, source)

	# So now we have len(source) >= len(target).
	if len(target) == 0:
		return len(source)

	# We call tuple() to force strings to be used as sequences
	# ('c', 'a', 't', 's') - numpy uses them as values by default.
	source = np.array(tuple(source))
	target = np.array(tuple(target))

	# We use a dynamic programming algorithm, but with the
	# added optimization that we only need the last two rows
	# of the matrix.
	previous_row = np.arange(target.size + 1)
	for s in source:
		# Insertion (target grows longer than source):
		current_row = previous_row + 1

		# Substitution or matching:
		# Target and source items are aligned, and either
		# are different (cost of 1), or are the same (cost of 0).
		current_row[1:] = np.minimum(
				current_row[1:],
				np.add(previous_row[:-1], target != s))

		# Deletion (target grows shorter than source):
		current_row[1:] = np.minimum(
				current_row[1:],
				current_row[0:-1] + 1)

		previous_row = current_row

	return previous_row[-1]

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


if __name__ == "__main__":
	
	with open('Dataset-Treino-Anonimizado-3.json') as data_file:
		dataSet = json.load(data_file)

	# a = 22
	# b = 436
	# print toClean(dataSet[a]['title']+" "+dataSet[a]['description'])
	# print "------------------------"
	# print toClean (dataSet[b]['title']+" "+dataSet[b]['description'])

	# print levenshtein( toClean(dataSet[a]['title']+" "+dataSet[a]['description']),toClean(dataSet[b]['title']+" "+dataSet[b]['description'] ) )
	
	grafo = {}
	setVisitados = set()
	for vaga1 in dataSet:
		textoVaga1 = toClean(vaga1['title']+" "+vaga1['description'])
		
		grafo[int(vaga1['id'])] = set()

		if vaga1['id'] not in setVisitados:
			for vaga2 in dataSet:
				if vaga1['id']!= vaga2['id']:
					textoVaga2 = toClean(vaga2['title']+" "+vaga2['description'])

					d = levenshtein(textoVaga1,textoVaga2)
					valor = 1 - d/float(max( len(textoVaga2),len(textoVaga1)) )
					if valor==1.0:
						setVisitados.add(vaga1['id'])
						setVisitados.add(vaga2['id'])
					if valor>0.6:
						grafo[int(vaga1['id'])].add(int(vaga2['id']))

						print "distancia entre ",vaga1['id']," ",vaga2['id'], 'd ', valor

	itemGrupo = dfs_paths(grafo)

	with open("gabarito.csv", "wb") as csv_file:
		writer = csv.writer(csv_file, delimiter=',')
		for vaga in dataSet:
			ID = int(vaga['id'])
			# print ID," ",itemGrupo[ID], "\n"
			writer.writerow([ID, itemGrupo[ID]])

