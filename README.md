# Estudos · Studies

Laboratório de estudos em Python e dados: automação, ETL, nuvem, OCR e NLP.
Cada pasta é um exercício independente, com um README curto do que foi praticado.

*Python and data study lab: automation, ETL, cloud, OCR and NLP. Each folder is a
self-contained exercise with a short README of what was practiced.*

O projeto completo, que junta tudo isto numa plataforma de dados de ponta a ponta,
está em **[plataforma-dados-pessoais](https://github.com/alexandremanzo/plataforma-dados-pessoais)**.

*The full project, which brings all of this together into an end-to-end data
platform, lives in **[plataforma-dados-pessoais](https://github.com/alexandremanzo/plataforma-dados-pessoais)**.*

## Índice · Index

| Pasta · Folder | Tema | Topic | Stack |
| --- | --- | --- | --- |
| [etl-extrato-excel](etl-extrato-excel/) | Limpeza e categorização de extrato de corretora | Cleaning and categorizing a brokerage statement | pandas, numpy, regex |
| [dados-abertos-cnpj](dados-abertos-cnpj/) | Download e extração dos dados abertos de CNPJ da Receita Federal | Downloading and extracting Brazil's open company registry | requests, zipfile, Selenium |
| [bigquery](bigquery/) | Consulta ao Google BigQuery com conta de serviço | Querying Google BigQuery with a service account | google-cloud-bigquery, pandas |
| [ocr-bioimpedancia](ocr-bioimpedancia/) | OCR de laudo de bioimpedância para planilha | OCR of a body-composition report into a spreadsheet | pytesseract, Pillow, regex, openpyxl |
| [analise-sentimento](analise-sentimento/) | Classificação de sentimento com Transformers | Sentiment classification with Transformers | Hugging Face transformers, pandas |
| [automacao](automacao/) | Automação de navegador e alerta por e-mail | Browser automation and e-mail alerts | Selenium, PyAutoGUI, smtplib |
| [sql-server](sql-server/) | Instalação, conexão e camadas medalhão | Setup, connection and medallion layers | SQL Server, pyodbc, SQLAlchemy |
| [ciencia-dados-credito](ciencia-dados-credito/) | Scores, métricas, políticas e monitoramento de modelos de crédito | Credit scoring, model metrics, policies and monitoring | pandas, scikit-learn, SQL |
| [ambiente-python](ambiente-python/) | Ambiente virtual e dependências | Virtual environments and dependencies | venv, pip |

## Como rodar · How to run

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (source venv/bin/activate no Linux/macOS)
pip install -r ambiente-python/requirements.txt
```

Credenciais e caminhos de arquivo vêm de variáveis de ambiente (por exemplo
`GOOGLE_APPLICATION_CREDENTIALS`) e nunca são versionados.

*Credentials and file paths come from environment variables (e.g.
`GOOGLE_APPLICATION_CREDENTIALS`) and are never committed.*
