--------------------------------------------------------
# Passos para criar um ambiente virtual no Python
--------------------------------------------------------
## 1. Abra o terminal e vá até a pasta do seu projeto:
--------------------------------------------------------
cd /caminho/do/seu/projeto

--------------------------------------------------------
## 2. Crie o ambiente virtual:
--------------------------------------------------------
python -m venv venv

--------------------------------------------------------
## 3. Ative o ambiente virtual:
--------------------------------------------------------
# 1. para habilitar as permissões de scripts
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass 
# 2. para ativar o ambiente virtual
venv\Scripts\activate

--------------------------------------------------------
## 4. Instalar pacotes
--------------------------------------------------------
pip install nome-do-pacote

--------------------------------------------------------
## 5. Desativar o ambiente virtual:
--------------------------------------------------------
Quando terminar:

deactivate
--------------------------------------------------------
## 6. Dica extra: Salvar dependências
--------------------------------------------------------

Para salvar os pacotes instalados:
pip freeze > requirements.txt

E para instalar tudo depois:
pip install -r requirements.txt