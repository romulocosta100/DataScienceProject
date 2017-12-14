import json

with open('vagas/Dataset-Treino-Anonimizado-3.json') as data_file:
		dataSet = json.load(data_file)


#

with open('5000vagas.json', 'w') as f:
	json.dump(dataSet[:5000], f)


