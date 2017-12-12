## exibe o help
all: help

# See https://gist.github.com/prwhite/8168133#comment-1313022
## comandos do projeto
help:
	@echo
	@printf "Targets available:\n\n"
	@awk '/^[a-zA-Z\-\_0-9]+:/ { \
		helpMessage = match(lastLine, /^## (.*)/); \
		if (helpMessage) { \
			helpCommand = substr($$1, 0, index($$1, ":")); \
			helpMessage = substr(lastLine, RSTART + 3, RLENGTH); \
			printf "%-35s%s\n", helpCommand, helpMessage; \
		} \
	} \
	{ lastLine = $$0 }' $(MAKEFILE_LIST)
	@echo

## atualiza os pacotes usando o requirements.txt
requirements:
	@echo 'Instalando ./requirements.txt'
	@pip install -r ./requirements.txt
