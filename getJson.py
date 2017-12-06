import json

with open('Dataset-Treino-Anonimizado-3.json') as data_file:
		dataSet = json.load(data_file)


#

with open('100vagas.json', 'w') as f:
	json.dump(dataSet[:100], f)


