# SQL Server

## Instalacao

- **Edicao:** Microsoft SQL Server 2022 Developer (64-bit) — 16.0.1000.6
- **Instancia:** `MSSQLSERVER` (instancia padrao)
- **Servidor:** `localhost`
- **Data root:** `C:\Program Files\Microsoft SQL Server\MSSQL16.MSSQLSERVER\MSSQL`
- **Driver ODBC:** ODBC Driver 17 for SQL Server

## Conexao

> As credenciais **nao ficam neste arquivo**. Elas vivem em
> `.env` na raiz do projeto, que nunca e versionado.
> Use `.env.example` como modelo.

| Item | Onde esta |
| --- | --- |
| Servidor, banco, driver | `.env` → `DB_SERVER`, `DB_DATABASE`, `DB_DRIVER` |
| Usuario e senha | `.env` → `DB_USER`, `DB_PASSWORD` |
| Tipo de autenticacao | `.env` → `DB_AUTH` (`sql` ou `windows`) |

Nos scripts Python, nunca monte a connection string manualmente:

```python
from scripts.conf.config_db import get_connection, get_engine, BRONZE, SILVER, GOLD

with get_connection() as conn:
    ...

df.to_sql("itau_extrato", get_engine(), schema=BRONZE, if_exists="append", index=False)
```

## Bancos

| Banco | Papel |
| --- | --- |
| `db-manzo` | Banco central do projeto. Contem as camadas `bronze`, `silver` e `gold`. |
| `db-manzo-stage` | Legado. Sera descontinuado apos a migracao para as novas camadas. |

## Arquitetura de camadas (medalhao)

As tres camadas sao **schemas dentro de `db-manzo`**, nao bancos separados:

| Schema | Conteudo |
| --- | --- |
| `bronze` | Copia fiel dos arquivos de origem, sem transformacao. Colunas em texto + metadados de linhagem. |
| `silver` | Dados limpos, tipados, padronizados e deduplicados. Uma tabela por entidade de negocio. |
| `gold` | Star schema pronto para consumo. **Unica camada que o Power BI consulta.** |

A descricao de cada camada tambem esta gravada no proprio banco
(`MS_Description`), alimentando o glossario de dados do projeto.

Script de criacao: `scripts/sql/01_cria_camadas.sql` do projeto [plataforma-dados-pessoais](https://github.com/alexandremanzo/plataforma-dados-pessoais) (idempotente).
