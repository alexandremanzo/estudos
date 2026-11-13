import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from google.cloud import bigquery
import pandas as pd
import os

# Configurações de e-mail (recomendado usar variáveis de ambiente)
EMAIL_ORIGEM = os.getenv("EMAIL_ORIGEM", "seu@email.com")
EMAIL_DESTINO = os.getenv("EMAIL_DESTINO", "destinatario@email.com")
SENHA = os.getenv("EMAIL_SENHA", "sua_senha_de_app")
SMTP_SERVER = "smtp.gmail.com"
PORTA = 587

def enviar_email(assunto, mensagem):
    msg = MIMEMultipart()
    msg["From"] = EMAIL_ORIGEM
    msg["To"] = EMAIL_DESTINO
    msg["Subject"] = assunto

    msg.attach(MIMEText(mensagem, "plain"))

    with smtplib.SMTP(SMTP_SERVER, PORTA) as server:
        server.starttls()
        server.login(EMAIL_ORIGEM, SENHA)
        server.send_message(msg)

def executar_consulta():
    try:
        client = bigquery.Client()
        query = "SELECT nome, idade FROM `projeto.dataset.tabela` LIMIT 5"
        resultado = client.query(query).to_dataframe()

        mensagem = f"Consulta executada com sucesso!\n\n{resultado.to_string(index=False)}"
        enviar_email("✅ Consulta BigQuery bem-sucedida", mensagem)
        print("Email de sucesso enviado!")

    except Exception as e:
        mensagem = f"Erro ao executar a consulta:\n\n{str(e)}"
        enviar_email("❌ Erro na consulta BigQuery", mensagem)
        print("Email de erro enviado!")

if __name__ == "__main__":
    executar_consulta()
