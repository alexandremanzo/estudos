# ETL de extrato em Excel · Excel statement ETL

Lê o extrato exportado da corretora, remove cabeçalho e rodapé administrativos,
tipa os valores e categoriza cada lançamento (compra, venda, rendimento, IR,
transferência) com regras e regex, incluindo a extração do código do ativo.
Foi o protótipo das regras que hoje rodam na camada silver da plataforma.

*Reads a brokerage statement export, strips the administrative header and footer,
types the amounts and categorizes each entry (buy, sell, income, tax, transfer)
with rules and regex, including ticker extraction. It was the prototype of the
rules that now run in the platform's silver layer.*

`pandas` · `numpy` · `re` — entrada em `dados/extrato.xlsx` (não versionado).
