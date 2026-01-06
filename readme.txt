README – Configuração do Projeto (Python + Webots)

1) Criar o ambiente virtual (venv)
No diretório do projeto, execute:

python -m venv venv

2) Atualizar o pip
Após criar o venv, atualize o pip:

venv\Scripts\python -m pip install --upgrade pip    (Windows)
venv/bin/python -m pip install --upgrade pip        (Linux / macOS)

3) Instalar as dependências
Instale os pacotes do projeto:

venv\Scripts\python -m pip install -r requirements.txt    (Windows)
venv/bin/python -m pip install -r requirements.txt        (Linux / macOS)

4) Configurar o Python no Webots
No Webots, altere o caminho do interpretador Python para:

<diretório_do_projeto>/venv/Scripts/python.exe    (Windows)
<diretório_do_projeto>/venv/bin/python            (Linux / macOS)

Pronto. O Webots passará a usar o Python do ambiente virtual.
