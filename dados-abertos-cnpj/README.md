# Dados abertos de CNPJ · Brazil open company data

Monta a URL do mês corrente, baixa os arquivos públicos da Receita Federal
(Empresas, Estabelecimentos, Sócios, CNAEs…), extrai cada ZIP numa pasta por mês
e trata falhas de download. Primeiro com Selenium, depois só com `requests`.

*Builds the current month's URL, downloads Brazil's public company registry files
(companies, establishments, partners, activity codes…), extracts each ZIP into a
per-month folder and handles download failures. First with Selenium, then with
plain `requests`.*

`requests` · `zipfile` · `selenium`
