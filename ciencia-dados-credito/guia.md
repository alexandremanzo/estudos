# Ciência de dados aplicada a crédito em cooperativas

Guia de estudo com os temas do dia a dia de um cientista de dados numa
cooperativa financeira: modelos de score e propensão, métricas de validação,
políticas de crédito e cobrança, monitoramento e governança.

Cada tópico traz três partes:

- **O que é:** uma descrição curta e objetiva.
- **Na cooperativa:** um caso prático com cooperados, crédito, cobrança ou relacionamento.
- **Na prática:** códigos numerados (por exemplo, *Código 2.2*), cada um com
  título, o que faz, o que recebe e o que devolve. Os passos vêm comentados
  dentro do código.

### Como usar os códigos

- Os códigos em **Python** rodam **em sequência**, num mesmo notebook ou
  script. A seção 0 gera uma base fictícia, e os blocos seguintes reaproveitam
  as variáveis dos anteriores (`df`, `treino`, `teste`, `modelo`...).
- Os códigos em **SQL** (T-SQL, SQL Server) usam tabelas ilustrativas
  (`cooperado`, `contrato`, `posicao_mensal`, `score_historico`). Eles mostram
  o formato da consulta e precisam ser adaptados ao seu banco.
- Todos os dados são **fictícios**. Os números mudam se a semente aleatória
  mudar, mas as conclusões não.

```bash
pip install pandas numpy scikit-learn
```

## Sumário

0. [Base de exemplo](#0-base-de-exemplo)
1. [Ciclo de vida de um modelo](#1-ciclo-de-vida-de-um-modelo)
2. [Modelos de score e propensão](#2-modelos-de-score-e-propensão)
3. [Técnicas estatísticas, econométricas e de machine learning](#3-técnicas-estatísticas-econométricas-e-de-machine-learning)
4. [Base analítica](#4-base-analítica)
5. [Preparação dos dados](#5-preparação-dos-dados)
6. [Métricas de desempenho e estabilidade](#6-métricas-de-desempenho-e-estabilidade)
7. [Pontos de corte, faixas de risco e regras de decisão](#7-pontos-de-corte-faixas-de-risco-e-regras-de-decisão)
8. [Análise de carteira: vintage, MOB, rolagem e curvas de inadimplência](#8-análise-de-carteira-vintage-mob-rolagem-e-curvas-de-inadimplência)
9. [Segmentação de cooperados](#9-segmentação-de-cooperados)
10. [Políticas, Champion-Challenger e estudos de impacto](#10-políticas-champion-challenger-e-estudos-de-impacto)
11. [Monitoramento, backtesting e drift](#11-monitoramento-backtesting-e-drift)
12. [Governança de dados e de modelos](#12-governança-de-dados-e-de-modelos)
13. [Indicadores, dashboards e comunicação executiva](#13-indicadores-dashboards-e-comunicação-executiva)
- [Índice dos códigos](#índice-dos-códigos)
- [Glossário rápido](#glossário-rápido)

---

## 0. Base de exemplo

Carteira fictícia de 20 mil cooperados com crédito ativo, fotografados em
safras mensais de 2024. O alvo `mau` vale 1 quando o cooperado atingiu
**mais de 90 dias de atraso nos 12 meses seguintes**. É a definição de
inadimplência mais usada em modelos de crédito.

| Coluna | Significado |
|---|---|
| `id_cooperado` | Identificador do cooperado |
| `safra` | Mês da foto (data de referência) |
| `idade`, `renda` | Cadastro; a renda tem nulos, como no mundo real |
| `meses_associado` | Tempo de cooperativa |
| `qtd_produtos` | Quantos produtos usa (conta, cartão, consórcio, seguro...) |
| `uso_limite` | Fração do limite rotativo em uso (0 a 1) |
| `atraso_max_6m` | Maior atraso, em dias, nos 6 meses anteriores |
| `saldo_aplicacao` | Saldo em RDC e poupança |
| `regiao` | Capital, Interior ou Rural |
| `mau` | **Alvo:** 1 se passou de 90 dias de atraso nos 12 meses seguintes |

**Código 0.1 · Gerar a carteira fictícia de cooperados**

- **O que faz:** cria a base usada em todo o guia, com um comportamento de risco "verdadeiro" que os modelos vão tentar descobrir.
- **Saída:** o DataFrame `df`, com 20 mil linhas e as colunas da tabela acima.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)          # semente fixa: mesmo resultado a cada execução
n = 20_000

# 1) Características dos cooperados
df = pd.DataFrame({
    "id_cooperado": np.arange(1, n + 1),
    "safra": rng.choice(pd.period_range("2024-01", "2024-12", freq="M").astype(str), n),
    "idade": rng.integers(18, 80, n),
    "renda": rng.lognormal(8.2, 0.6, n).round(2),             # renda mensal (R$)
    "meses_associado": rng.integers(1, 240, n),
    "qtd_produtos": rng.poisson(2.5, n),
    "uso_limite": rng.beta(2, 5, n).round(3),
    "atraso_max_6m": rng.choice([0, 5, 15, 30, 60], n, p=[.70, .12, .09, .06, .03]),
    "saldo_aplicacao": rng.exponential(5_000, n).round(2),
    "regiao": rng.choice(["Capital", "Interior", "Rural"], n, p=[.4, .4, .2]),
})

# 2) Regra "verdadeira" de risco (na vida real ninguém a conhece; o modelo tenta estimá-la)
#    usar muito o limite e ter atrasado aumentam o risco;
#    tempo de casa, produtos e aplicações reduzem.
logit = (-2.2 + 3.0 * df.uso_limite + 0.035 * df.atraso_max_6m
         - 0.004 * df.meses_associado - 0.25 * df.qtd_produtos
         - 0.00004 * df.saldo_aplicacao.clip(upper=50_000)
         + rng.normal(0, 0.5, n))
df["mau"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

# 3) Renda sem preenchimento: mais comum em cadastro desatualizado (e de risco maior)
df.loc[rng.random(n) < np.where(df.mau == 1, 0.10, 0.045), "renda"] = np.nan

print(f"{len(df):,} cooperados · taxa de maus {df.mau.mean():.1%}")
```

---

## 1. Ciclo de vida de um modelo

**O que é:** a sequência que todo modelo percorre.
1. **Planejar:** o problema, o alvo e o público.
2. **Desenvolver:** base, variáveis e algoritmo.
3. **Validar:** desempenho fora da amostra e revisão independente.
4. **Implantar:** o modelo entra no sistema de decisão.
5. **Monitorar:** desempenho e estabilidade ao longo do tempo.

O ciclo recomeça quando o monitoramento mostra perda de desempenho.

**Na cooperativa:**

- **Exemplo 1, renovação de limite:** a área de crédito pede um score para renovar automaticamente o limite do cheque especial.
  - O alvo é ter atraso acima de 90 dias em 12 meses.
  - O público são os cooperados com conta há mais de 6 meses.
  - O modelo é desenvolvido com as safras de 2023 e 2024 e validado de forma independente (KS e PSI).
  - Depois de implantado, o score é calculado todo mês e um painel acompanha KS e PSI a cada fechamento.
- **Exemplo 2, campanha de consórcio:** o marketing quer saber a quem oferecer consórcio.
  - O ciclo é o mesmo, mas a validação olha *lift* em vez de KS.
  - O monitoramento acompanha a conversão real de cada campanha.
- **Exemplo 3, mudança de cenário:** a taxa Selic sobe e a inadimplência da carteira aumenta.
  - O monitoramento mostra a PD prevista abaixo da realizada.
  - O ciclo volta à etapa de desenvolvimento para recalibrar o modelo (seção 11).

| Etapa | Pergunta-chave | Entregável |
|---|---|---|
| Planejar | Que decisão o modelo vai apoiar? | Termo de abertura: alvo, público, janelas |
| Desenvolver | Quais variáveis explicam o alvo? | Base analítica, modelo treinado |
| Validar | Funciona em dados que não viu? | Relatório com KS, Gini, PSI, aderência |
| Implantar | Como chega à decisão? | Score na esteira, regras e cut-offs |
| Monitorar | Continua funcionando? | Painel mensal, alertas de drift |

---

## 2. Modelos de score e propensão

Todos os modelos desta seção estimam **a probabilidade de um evento**.
O que muda entre eles é o evento, o público e o momento da decisão.

| Modelo | Evento previsto | Público | Decisão que apoia |
|---|---|---|---|
| **Credit Score** (aplicação) | Atrasar > 90 dias | Proponentes, inclusive novos cooperados | Aprovar e precificar a concessão |
| **Behavior Score** | Atrasar > 90 dias | Cooperados com histórico na cooperativa | Limite, renovação, pré-aprovado |
| **Collection Score** | Pagar ou se regularizar | Cooperados já em atraso | Régua e canal de cobrança |
| **Propensão / cross-sell** | Contratar um produto | Base ativa | Campanhas e ofertas |
| **Retenção (churn)** | Encerrar o relacionamento | Base ativa | Ações de retenção |
| **Inadimplência / atraso** | Atrasar a próxima parcela | Contratos ativos | Cobrança preventiva |
| **Recuperação** | % do saldo recuperado | Contratos em prejuízo | Acordo, desconto, cessão |

### 2.1 Credit Score (aplicação)

**O que é:** o score da **concessão**. Usa dados do momento da proposta:
cadastro, renda, birôs externos e, quando o proponente já é cooperado, o
relacionamento. É a principal ferramenta para quem tem pouco histórico interno.

**Na cooperativa:**
- **Novo associado:** pede R$ 15 mil de crédito pessoal. Sem histórico interno, o score combina score de birô, renda declarada e comprometimento da parcela, e decide entre aprovar, aprovar com garantia ou negar.
- **Cooperado antigo:** o mesmo pedido de quem já é cooperado ganha pontos pelo relacionamento. Ser cooperado entra como variável.

**Código 2.1 · Credit score de aplicação e decisão de uma proposta**

- **O que faz:** treina um score com dados de proposta e avalia duas propostas novas.
- **Entrada:** propostas fictícias com score de birô, renda, valor solicitado e se o proponente já é cooperado.
- **Saída:** a probabilidade de inadimplência e a decisão de cada proposta.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 1) Histórico fictício de propostas aprovadas e o desempenho delas
rng_ap = np.random.default_rng(7)
m = 8_000
propostas = pd.DataFrame({
    "score_bureau": rng_ap.integers(300, 1000, m),
    "renda_declarada": rng_ap.lognormal(8.0, 0.6, m).round(2),
    "valor_solicitado": rng_ap.choice([5_000, 10_000, 15_000, 30_000], m),
    "ja_cooperado": rng_ap.integers(0, 2, m),
})
# comprometimento = parcela (24x, sem juros para simplificar) / renda
propostas["comprometimento"] = (propostas.valor_solicitado / 24) / propostas.renda_declarada
logit_ap = (-2.2 - 0.006 * (propostas.score_bureau - 650)
            + 2.5 * propostas.comprometimento - 0.5 * propostas.ja_cooperado)
propostas["mau"] = (rng_ap.random(m) < 1 / (1 + np.exp(-logit_ap))).astype(int)

# 2) Treina o modelo de aplicação
vars_ap = ["score_bureau", "comprometimento", "ja_cooperado"]
modelo_ap = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
modelo_ap.fit(propostas[vars_ap], propostas.mau)

# 3) Avalia duas propostas novas de R$ 15 mil
novas = pd.DataFrame({
    "proponente": ["novo associado", "cooperado há 8 anos"],
    "score_bureau": [610, 610],
    "comprometimento": [(15_000 / 24) / 3_500] * 2,       # renda de R$ 3.500
    "ja_cooperado": [0, 1],
})
novas["pd"] = modelo_ap.predict_proba(novas[vars_ap])[:, 1]
novas["decisao"] = pd.cut(novas.pd, [0, .12, .20, 1],      # faixas de PD da política
                          labels=["Aprova", "Aprova com garantia", "Nega"])
print(novas[["proponente", "pd", "decisao"]].round(3))
```

### 2.2 Behavior Score

**O que é:** o score do **comportamento**. Usa o histórico do cooperado na
própria cooperativa: uso de limite, atrasos, aplicações e tempo de casa. É
recalculado todo mês e costuma ser mais preciso que o de aplicação.

**Na cooperativa:**
- **Pré-aprovado:** todo mês, o behavior score decide quem recebe crédito pré-aprovado no aplicativo.
- **Limite do cartão:** também decide quem tem o limite aumentado e quem tem o limite reduzido preventivamente.
- **Gerente:** recebe a lista dos cooperados que mais pioraram de score no mês, para conversar antes de virar atraso.

**Código 2.2 · Treinar o behavior score e converter em pontos**

- **O que faz:** separa treino e teste, treina uma regressão logística e converte a probabilidade em uma escala de pontos.
- **Entrada:** a base `df` da seção 0.
- **Saída:** `modelo` e `teste`, com as colunas `pd_mau` (probabilidade) e `score` (pontos). Os dois são reusados no resto do guia.

A escala usa a convenção *PDO* (*points to double the odds*). Com 600
pontos, a chance é de 10 bons para cada mau, e cada 20 pontos a mais
dobram essa chance.

```python
from sklearn.model_selection import train_test_split

variaveis = ["uso_limite", "atraso_max_6m", "meses_associado", "qtd_produtos", "saldo_aplicacao"]

# 1) Separa 70% para treinar e 30% para testar, mantendo a mesma taxa de maus nos dois
treino, teste = train_test_split(df, test_size=0.3, random_state=42, stratify=df.mau)

# 2) Padroniza as variáveis e treina a regressão logística
modelo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
modelo.fit(treino[variaveis], treino.mau)

# 3) Converte probabilidade em pontos
def para_score(pd_mau, base=600, odds_base=10, pdo=20):
    """Converte PD em score: 600 pontos = 10 bons para 1 mau; +20 pontos dobram a chance."""
    fator = pdo / np.log(2)
    offset = base - fator * np.log(odds_base)
    return np.round(offset + fator * np.log((1 - pd_mau) / pd_mau)).astype(int)

teste = teste.assign(pd_mau=modelo.predict_proba(teste[variaveis])[:, 1])
teste["score"] = para_score(teste.pd_mau)
print(teste[["id_cooperado", "pd_mau", "score"]].head())
```

**Código 2.3 · Ler os pesos do modelo (o que mais pesa no score)**

- **O que faz:** mostra o coeficiente de cada variável. Em variáveis padronizadas, o tamanho indica a importância e o sinal indica o sentido.
- **Para que serve:** checar se o modelo faz sentido de negócio. Por exemplo, mais aplicação tem de reduzir o risco.

```python
coef = pd.Series(modelo[-1].coef_[0], index=variaveis).sort_values()
print(coef.round(3))
# negativo = reduz o risco (bom); positivo = aumenta o risco
```

### 2.3 Collection Score

**O que é:** o score da **cobrança**. O público são cooperados já em atraso,
e o evento é a cura, isto é, voltar a ficar em dia num prazo (por exemplo,
30 dias). Serve para concentrar o esforço onde a cobrança faz diferença.

**Na cooperativa:** 1.200 cooperados estão entre 5 e 30 dias de atraso.
- **Score alto:** muitos se regularizam sozinhos e recebem só SMS e lembrete no aplicativo.
- **Score médio:** a ligação do gerente faz diferença.
- **Score baixo:** vão direto para renegociação. Esperar só aumenta a perda.

**Código 2.4 · Collection score e régua de cobrança**

- **O que faz:** treina um modelo de probabilidade de cura entre os cooperados em atraso e distribui cada um numa ação da régua.
- **Entrada:** os cooperados de `df` com `atraso_max_6m > 0`.
- **Saída:** a tabela `cobranca`, com `p_cura` e `acao`, e o resumo da taxa de cura real por ação.

```python
# 1) Público: só quem está em atraso; evento: regularizou em até 30 dias
cobranca = df[df.atraso_max_6m > 0].copy()
p_cura = 1 / (1 + np.exp(-(1.5 - 0.04 * cobranca.atraso_max_6m
                            + 0.00008 * cobranca.saldo_aplicacao.clip(upper=30_000)
                            - 2.0 * cobranca.uso_limite)))
cobranca["curou_30d"] = (rng.random(len(cobranca)) < p_cura).astype(int)

# 2) Treina o modelo de cura
vars_cob = ["atraso_max_6m", "saldo_aplicacao", "uso_limite", "meses_associado"]
modelo_cob = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
modelo_cob.fit(cobranca[vars_cob], cobranca.curou_30d)
cobranca["p_cura"] = modelo_cob.predict_proba(cobranca[vars_cob])[:, 1]

# 3) Régua: quanto maior a chance de cura, mais barato o canal
cobranca["acao"] = pd.cut(cobranca.p_cura, [0, .4, .7, 1],
                          labels=["Renegociação", "Ligação do gerente", "SMS / app"])
print(cobranca.groupby("acao", observed=True).agg(cooperados=("p_cura", "size"),
                                                  cura_real=("curou_30d", "mean")))
```

**Código 2.5 · Custo da régua de cobrança**

- **O que faz:** compara o custo de ligar para todos os cooperados em atraso com o custo da régua por score.
- **Custos fictícios:** SMS custa R$ 0,10, uma ligação R$ 6 e uma renegociação R$ 25.

```python
custo = {"SMS / app": 0.10, "Ligação do gerente": 6.00, "Renegociação": 25.00}
custo_regua = cobranca.acao.map(custo).astype(float).sum()
custo_todos_ligacao = len(cobranca) * custo["Ligação do gerente"]
print(f"ligar para todos: R$ {custo_todos_ligacao:,.0f}   régua por score: R$ {custo_regua:,.0f}")
```

### 2.4 Propensão à contratação e cross-sell

**O que é:** a probabilidade de o cooperado **contratar** um produto num
prazo. Mede-se pelo **lift**: quantas vezes o grupo escolhido pelo modelo
converte mais que a base.

**Na cooperativa:**
- **Consórcio:** a campanha tem orçamento para 2 mil contatos. Em vez de sortear, a cooperativa contata os 10% com maior propensão. Se esse decil converte 3 vezes a média, o custo por venda cai a um terço.
- **Seguro de vida:** cooperados entre 30 e 50 anos, com filhos dependentes no cadastro e crédito imobiliário, recebem a oferta no momento da contratação do financiamento.

**Código 2.6 · Propensão a consórcio e lift por decil**

- **O que faz:** treina um gradient boosting para prever a contratação de consórcio e mede o lift de cada decil de propensão.
- **Saída:** o lift por decil. O decil 1 são os 10% com maior propensão.

```python
from sklearn.ensemble import HistGradientBoostingClassifier

# 1) Evento fictício: contratou consórcio em 90 dias
p_consorcio = 1 / (1 + np.exp(-(-3.5 + 0.00012 * df.saldo_aplicacao.clip(upper=30_000)
                                + 0.3 * df.qtd_produtos + 0.01 * (df.idade - 40).abs() * -1)))
df["contratou_consorcio"] = (rng.random(n) < p_consorcio).astype(int)

# 2) Treina o modelo de propensão
vars_prop = ["saldo_aplicacao", "qtd_produtos", "idade", "meses_associado", "uso_limite"]
tr, te = train_test_split(df, test_size=0.3, random_state=1, stratify=df.contratou_consorcio)
prop = HistGradientBoostingClassifier(max_depth=3, random_state=1).fit(tr[vars_prop], tr.contratou_consorcio)
te = te.assign(p=prop.predict_proba(te[vars_prop])[:, 1])

# 3) Lift: conversão de cada decil ÷ conversão média
te["decil"] = pd.qcut(te.p.rank(method="first", ascending=False), 10, labels=range(1, 11))
lift = te.groupby("decil", observed=True).contratou_consorcio.mean() / te.contratou_consorcio.mean()
print(lift.round(2))
```

**Código 2.7 · Retorno da campanha: sortear ou usar o modelo**

- **O que faz:** compara as vendas de 2 mil contatos sorteados com as de 2 mil contatos escolhidos pela propensão. Os números são proporcionais ao conjunto de teste.

```python
contatos = 2_000
conv_media = te.contratou_consorcio.mean()
conv_top = te.nlargest(contatos, "p").contratou_consorcio.mean()
print(f"sorteio: {contatos * conv_media:.0f} vendas   modelo: {contatos * conv_top:.0f} vendas")
```

### 2.5 Retenção (churn)

**O que é:** a probabilidade de o cooperado **reduzir ou encerrar** o
relacionamento: portabilidade de salário, resgate das aplicações,
encerramento da conta. O alvo precisa ser bem definido, por exemplo: "resgatou
mais de 80% das aplicações e zerou a movimentação em 60 dias".

**Na cooperativa:**
- **Salário portado:** cooperados com salário portado e queda de movimentação recebem a visita do gerente antes de levarem o salário para outro banco.
- **Prioridade:** a lista é ordenada por **risco × valor**. Perder um cooperado com R$ 80 mil aplicados pesa mais que perder um com conta zerada. Reter um cooperado costuma custar menos que conquistar outro.

**Código 2.8 · Modelo de churn e lista priorizada para os gerentes**

- **O que faz:** estima a probabilidade de saída em 90 dias e monta a lista dos 10 cooperados com maior valor em risco.
- **Saída:** a lista com `p_churn`, o saldo aplicado e o valor em risco (`p_churn × saldo`).

```python
# 1) Sinais de afastamento (fictícios)
rng_ch = np.random.default_rng(11)
rel = df[["id_cooperado", "qtd_produtos", "meses_associado", "saldo_aplicacao"]].copy()
rel["queda_movimentacao"] = rng_ch.beta(2, 6, n)            # % de queda nos últimos 3 meses
rel["salario_portado"] = (rng_ch.random(n) < 0.35).astype(int)
logit_ch = (-3.0 + 4.0 * rel.queda_movimentacao - 0.3 * rel.qtd_produtos
            + 0.8 * rel.salario_portado - 0.003 * rel.meses_associado)
rel["saiu_90d"] = (rng_ch.random(n) < 1 / (1 + np.exp(-logit_ch))).astype(int)

# 2) Treina o modelo de churn
vars_ch = ["queda_movimentacao", "qtd_produtos", "salario_portado", "meses_associado"]
modelo_ch = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
modelo_ch.fit(rel[vars_ch], rel.saiu_90d)
rel["p_churn"] = modelo_ch.predict_proba(rel[vars_ch])[:, 1]

# 3) Prioridade = probabilidade de sair × quanto o cooperado tem na cooperativa
rel["valor_em_risco"] = rel.p_churn * rel.saldo_aplicacao
print(rel.nlargest(10, "valor_em_risco")[["id_cooperado", "p_churn", "saldo_aplicacao", "valor_em_risco"]].round(2))
```

### 2.6 Propensão a atraso e inadimplência

**O que é:** um modelo de curto prazo que prevê o atraso da **próxima
parcela**. Fica entre o behavior score e o collection score e permite agir
**antes** do vencimento.

**Na cooperativa:**
- **Lembrete preventivo:** cinco dias antes do vencimento, quem tem alta propensão a atrasar recebe um lembrete com o código de pagamento. Pequenos atrasos evitados reduzem a rolagem para faixas piores.
- **Débito automático:** a propensão também indica a quem oferecer débito automático na conta.

**Código 2.9 · Selecionar quem recebe o lembrete preventivo**

- **O que faz:** cruza a data de vencimento com a PD do behavior score e escolhe quem recebe o lembrete.
- **Entrada:** `teste`, com a PD do Código 2.2, e um dia de vencimento fictício.
- **Saída:** a quantidade de lembretes e a inadimplência do grupo selecionado comparada à da base.

```python
rng_venc = np.random.default_rng(5)
teste["dias_para_vencer"] = rng_venc.integers(0, 30, len(teste))

lembrete = teste[(teste.dias_para_vencer <= 5) & (teste.pd_mau >= 0.15)]
print(f"{len(lembrete)} lembretes · inadimplência do grupo {lembrete.mau.mean():.1%} "
      f"vs base {teste.mau.mean():.1%}")
```

### 2.7 Recuperação de crédito

**O que é:** estima **quanto** será recuperado de um contrato em prejuízo
(taxa de recuperação, ou 1 − LGD). É uma regressão, não uma classificação. A
previsão orienta o desconto máximo num acordo e a decisão de vender a
carteira.

**Na cooperativa:**
- **Acordo à vista:** se a recuperação esperada de um contrato é 25%, um acordo com 60% de desconto (recebendo 40% à vista) é vantajoso.
- **Cessão de carteira:** contratos com recuperação esperada abaixo de 5% são candidatos a venda, porque o custo de cobrar supera o que se recupera.

**Código 2.10 · Prever a recuperação e decidir sobre propostas de acordo**

- **O que faz:** treina uma regressão da taxa de recuperação e decide se cada proposta de acordo vale a pena.
- **Regra:** aceitar quando o valor à vista é maior que o valor esperado da cobrança, descontado um custo de 10% do recuperado.

```python
from sklearn.ensemble import HistGradientBoostingRegressor

# 1) Contratos em prejuízo (fictícios)
rng_rc = np.random.default_rng(13)
k = 3_000
prejuizo = pd.DataFrame({
    "saldo_devedor": rng_rc.lognormal(9, 0.7, k).round(2),
    "tem_garantia": (rng_rc.random(k) < 0.3).astype(int),
    "tem_aplicacao": (rng_rc.random(k) < 0.2).astype(int),
    "meses_em_prejuizo": rng_rc.integers(1, 36, k),
})
prejuizo["taxa_recuperacao"] = (0.15 + 0.35 * prejuizo.tem_garantia + 0.15 * prejuizo.tem_aplicacao
                                - 0.004 * prejuizo.meses_em_prejuizo
                                + rng_rc.normal(0, 0.08, k)).clip(0, 1)

# 2) Treina a regressão
vars_rc = ["saldo_devedor", "tem_garantia", "tem_aplicacao", "meses_em_prejuizo"]
modelo_rc = HistGradientBoostingRegressor(max_depth=3, random_state=13)
modelo_rc.fit(prejuizo[vars_rc], prejuizo.taxa_recuperacao)

# 3) Avalia propostas de acordo: o cooperado oferece pagar X% do saldo à vista
acordos = pd.DataFrame({
    "saldo_devedor": [12_000, 12_000], "tem_garantia": [0, 1],
    "tem_aplicacao": [0, 0], "meses_em_prejuizo": [18, 18],
    "oferta_pct": [0.20, 0.20],
})
acordos["recuperacao_esperada"] = modelo_rc.predict(acordos[vars_rc]).clip(0, 1)
acordos["valor_cobranca"] = acordos.recuperacao_esperada * (1 - 0.10)     # menos custo de cobrar
acordos["decisao"] = np.where(acordos.oferta_pct > acordos.valor_cobranca, "Aceita", "Contraproposta")
print(acordos[["tem_garantia", "oferta_pct", "recuperacao_esperada", "decisao"]].round(3))
```

---

## 3. Técnicas estatísticas, econométricas e de machine learning

**O que é:** o conjunto de algoritmos. Em crédito, **interpretabilidade e
estabilidade** valem tanto quanto acurácia, porque cada decisão precisa ser
explicável ao cooperado, à auditoria e ao regulador.

| Técnica | Uso típico | Vantagem | Cuidado |
|---|---|---|---|
| Regressão logística com WoE | Credit e behavior score | Explicável, estável, vira scorecard em pontos | Exige binning bem feito |
| Árvore de decisão | Segmentação e regras de política | Vira regra de negócio legível | Instável sozinha |
| Gradient boosting (LightGBM, XGBoost) | Propensão, cobrança | Mais poder preditivo | Precisa de explicabilidade (SHAP) |
| Análise de sobrevivência (Cox, Kaplan-Meier) | Tempo até o atraso ou o churn | Usa contratos ainda abertos | Interpretação menos direta |
| Regressão linear ou beta | Recuperação (LGD) | Prevê percentuais | Alvo limitado entre 0 e 1 |
| Séries temporais (ARIMA, regressão com macro) | Projeção de inadimplência da carteira | Liga carteira e cenário econômico | Poucas observações |

**Na cooperativa:**
- **Propensão a consórcio:** pode ser um gradient boosting, porque errar uma oferta custa pouco.
- **Score de concessão:** costuma ficar na regressão logística, porque cada ponto precisa ser explicado ao cooperado que teve o crédito negado.
- **Projeção de inadimplência para o orçamento:** pede uma série temporal ligada à Selic e ao desemprego.

**Código 3.1 · Comparar regressão logística e gradient boosting**

- **O que faz:** treina os dois algoritmos na mesma base e compara a AUC no mesmo teste.
- **Leitura:** se o ganho do boosting for pequeno, a logística costuma ganhar pela explicabilidade.

```python
from sklearn.metrics import roc_auc_score

gb = HistGradientBoostingClassifier(max_depth=3, random_state=42).fit(treino[variaveis], treino.mau)
print("AUC logística :", round(roc_auc_score(teste.mau, teste.pd_mau), 3))
print("AUC boosting  :", round(roc_auc_score(teste.mau, gb.predict_proba(teste[variaveis])[:, 1]), 3))
```

---

## 4. Base analítica

**O que é:** a tabela de modelagem, com uma linha por cooperado e por data de
referência. Ela define três coisas:
- a **janela de observação**: o passado, de onde saem as variáveis;
- a **janela de desempenho**: o futuro, onde se mede o alvo;
- a **definição de mau**.

O erro mais grave é o **vazamento**: uma variável que usa informação
posterior à data de referência.

**Na cooperativa:**
- **Janelas:** para cada cooperado em 30/06/2024, as variáveis vêm dos 6 meses anteriores e o alvo vem dos 12 meses seguintes.
- **Vazamento clássico:** o saldo devedor de julho é futuro, então não pode entrar como variável.
- **Vazamento disfarçado:** "teve renegociação" parece passado, mas se a renegociação foi feita *por causa* do atraso que se quer prever, o modelo aprende a resposta.

```text
        janela de observação           janela de desempenho
   |---------- 6 meses ----------|---------- 12 meses ----------|
 jan/24                       30/06/24                       jun/25
                            data de referência
   variáveis (X)                          alvo: atraso > 90 dias (y)
```

**Código 4.1 (SQL) · Montar a base analítica de uma data de referência**

- **O que faz:** para cada cooperado com 6 meses de histórico, calcula as variáveis no passado e o alvo no futuro, separados pela data de referência.
- **Saída:** uma linha por cooperado, com as variáveis e a coluna `mau`.

```sql
DECLARE @dt_ref date = '2024-06-30';

SELECT  c.id_cooperado,
        @dt_ref                                              AS dt_referencia,
        DATEDIFF(MONTH, c.dt_associacao, @dt_ref)            AS meses_associado,
        obs.atraso_max_6m,
        obs.uso_limite_medio_6m,
        CASE WHEN des.atraso_max_12m > 90 THEN 1 ELSE 0 END  AS mau
FROM cooperado AS c
-- passado: tudo que acontece até a data de referência vira variável
CROSS APPLY (
    SELECT MAX(p.dias_atraso)                                    AS atraso_max_6m,
           AVG(p.saldo_utilizado * 1.0 / NULLIF(p.limite, 0))    AS uso_limite_medio_6m
    FROM posicao_mensal AS p
    WHERE p.id_cooperado = c.id_cooperado
      AND p.dt_posicao >  DATEADD(MONTH, -6, @dt_ref)
      AND p.dt_posicao <= @dt_ref
) AS obs
-- futuro: usado SOMENTE para o alvo
CROSS APPLY (
    SELECT MAX(p.dias_atraso) AS atraso_max_12m
    FROM posicao_mensal AS p
    WHERE p.id_cooperado = c.id_cooperado
      AND p.dt_posicao >  @dt_ref
      AND p.dt_posicao <= DATEADD(MONTH, 12, @dt_ref)
) AS des
WHERE c.dt_associacao <= DATEADD(MONTH, -6, @dt_ref);   -- público com 6 meses de histórico
```

Empilhar várias datas de referência (as **safras**) dá volume ao modelo e
permite a validação **fora do tempo** (out-of-time, OOT): treina-se com as
safras antigas e testa-se com as recentes. É o teste mais próximo do uso
real, porque o modelo sempre vai escorar um futuro que não viu.

**Código 4.2 · Validação fora do tempo (out-of-time)**

- **O que faz:** treina com as safras de jan a set e testa com as de out a dez. Compara com a validação aleatória.
- **Leitura:** se a AUC cai muito no OOT, o modelo depende de algo que muda com o tempo.

```python
oot = df.safra >= "2024-10"
modelo_oot = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
modelo_oot.fit(df.loc[~oot, variaveis], df.loc[~oot, "mau"])

auc_dentro = roc_auc_score(teste.mau, teste.pd_mau)
auc_oot = roc_auc_score(df.loc[oot, "mau"], modelo_oot.predict_proba(df.loc[oot, variaveis])[:, 1])
print(f"AUC teste aleatório: {auc_dentro:.3f}   AUC fora do tempo: {auc_oot:.3f}")
```

---

## 5. Preparação dos dados

### 5.1 Análise exploratória (EDA)

**O que é:** conhecer a base antes de modelar. Olha-se distribuição,
nulos, valores impossíveis e a **taxa de maus por faixa** de cada variável.
A tendência dessa taxa indica se a variável tem relação com o risco.

**Na cooperativa:**
- **Uso do limite:** a taxa de maus sobe de forma consistente com o uso do limite. Isso confirma o que a área de crédito já intuía e dá confiança na variável.
- **Região:** a inadimplência no meio rural pode ter sazonalidade de safra agrícola, e isso pede variáveis próprias.

**Código 5.1 · Raio-X da base e taxa de maus por faixa**

- **O que faz:** resume as colunas, mostra o percentual de nulos e a taxa de maus por faixa de uso do limite e por região.

```python
# 1) Resumo estatístico das colunas numéricas
print(df.describe().T[["mean", "min", "50%", "max"]].round(1))

# 2) Percentual de nulos (só colunas com algum nulo)
print(df.isna().mean().loc[lambda s: s > 0])

# 3) Taxa de maus por faixa de uso do limite: deve crescer
df["faixa_uso"] = pd.cut(df.uso_limite, [0, .2, .4, .6, .8, 1], include_lowest=True)
print(df.groupby("faixa_uso", observed=True).mau.agg(cooperados="size", taxa_maus="mean"))

# 4) Taxa de maus por variável categórica
print(df.groupby("regiao").mau.mean().round(3))
```

### 5.2 Tratamento de missing

**O que é:** decidir o que fazer com o dado ausente. As opções são
imputar (mediana, valor de negócio), criar uma categoria "não informado" ou
descartar. Em crédito, **o próprio fato de faltar costuma ter informação**,
por isso vale guardar uma flag.

**Na cooperativa:**
- **Renda desatualizada:** cooperados sem renda atualizada são, em geral, os que não passam na agência há anos. A flag `renda_nao_informada` vira uma variável.
- **Sem birô:** a falta de score de birô indica alguém sem histórico de crédito no mercado ("thin file"). É outro perfil de risco, não um erro.

**Código 5.2 · Flag de ausência e imputação pela mediana**

- **O que faz:** cria a flag de renda não informada, preenche a renda com a mediana e compara a taxa de maus de quem tem e de quem não tem renda.

```python
df["renda_nao_informada"] = df.renda.isna().astype(int)        # guarda a informação da ausência
df["renda_trat"] = df.renda.fillna(df.renda.median())          # preenche para o modelo poder usar
print(df.groupby("renda_nao_informada").mau.mean().round(3))   # o nulo tem risco diferente?
```

### 5.3 Outliers

**O que é:** valores extremos que distorcem médias e coeficientes. Podem ser
erros (idade 150, renda R$ 1 bilhão) ou casos raros, mas verdadeiros. Erros
se corrigem. Casos legítimos se limitam por **winsorização**: cortar em
percentis, como 1% e 99%.

**Na cooperativa:**
- **Caso legítimo:** um produtor rural com R$ 2 milhões aplicados é real, mas não deve fazer o modelo achar que cada real a mais aplicado reduz o risco para sempre.
- **Erro de digitação:** uma renda de R$ 350.000 num cadastro de caixa de supermercado é provavelmente R$ 3.500,00 digitado sem a vírgula.

**Código 5.3 · Winsorização e regra do IQR**

- **O que faz:** limita o saldo aplicado entre os percentis 1 e 99 e conta quantas rendas ficam fora do intervalo interquartil (IQR).

```python
# 1) Winsorização: valores abaixo de p1 viram p1; acima de p99 viram p99
p01, p99 = df.saldo_aplicacao.quantile([.01, .99])
df["saldo_aplicacao_w"] = df.saldo_aplicacao.clip(p01, p99)

# 2) Regra do IQR: fora de [Q1 − 1,5·IQR, Q3 + 1,5·IQR] é candidato a outlier
q1, q3 = df.renda_trat.quantile([.25, .75])
iqr = q3 - q1
fora = (df.renda_trat < q1 - 1.5 * iqr) | (df.renda_trat > q3 + 1.5 * iqr)
print(f"rendas fora do intervalo IQR: {fora.mean():.1%}")
```

### 5.4 Discretização, WoE e IV

**O que é:** agrupar a variável em **faixas** (binning). Para cada faixa
calcula-se:
- **WoE** (*Weight of Evidence*): `ln(%bons / %maus)`. Mostra se a faixa é mais segura (positivo) ou mais arriscada (negativo) que a média.
- **IV** (*Information Value*): `Σ (%bons − %maus) × WoE`. Resume o poder da variável.

Faixas tornam o modelo estável, tratam outliers e nulos de uma vez e geram
um scorecard legível.

| IV | Poder preditivo |
|---|---|
| < 0,02 | Inútil |
| 0,02 – 0,10 | Fraco |
| 0,10 – 0,30 | Médio |
| 0,30 – 0,50 | Forte |
| > 0,50 | Suspeito: verificar vazamento |

**Na cooperativa:**
- **Atraso recente:** o `atraso_max_6m` aparece com IV alto, o que é esperado.
- **IV suspeito:** uma variável com IV de 0,9 pede investigação, porque pode estar medindo o próprio alvo.
- **Variável a revisar:** uma variável com IV abaixo de 0,02, como a idade nesta base, não ajuda a separar bons de maus e sai da lista.

**Código 5.4 · Função de WoE e IV**

- **O que faz:** divide a variável em faixas de mesmo tamanho e calcula maus, bons, WoE e IV de cada faixa.
- **Entrada:** a base, o nome da variável, o nome do alvo e o número de faixas.
- **Saída:** a tabela por faixa e o IV total.

```python
def woe_iv(base, var, alvo="mau", faixas=5):
    """Tabela de WoE por faixa e IV total de uma variável numérica."""
    # 1) Faixas por quantis (duplicates='drop' junta faixas iguais em variáveis discretas)
    faixa = pd.qcut(base[var], faixas, duplicates="drop")
    t = base.groupby(faixa, observed=True)[alvo].agg(maus="sum", total="count")
    t["bons"] = t.total - t.maus
    # 2) Distribuição de bons e de maus entre as faixas
    dist_b, dist_m = t.bons / t.bons.sum(), t.maus / t.maus.sum()
    # 3) WoE e contribuição de cada faixa para o IV
    t["woe"] = np.log(dist_b / dist_m)
    t["iv"] = (dist_b - dist_m) * t.woe
    return t, t.iv.sum()

tabela, iv = woe_iv(df, "uso_limite")
print(tabela.round(3), f"\nIV = {iv:.3f}")
```

**Código 5.5 · Ranking de variáveis por IV**

- **O que faz:** aplica a função a todas as candidatas e ordena da mais forte para a mais fraca.

```python
ranking_iv = {v: woe_iv(df, v)[1] for v in variaveis + ["idade", "renda_trat"]}
print(pd.Series(ranking_iv).sort_values(ascending=False).round(3))
```

### 5.5 Engenharia de atributos (feature engineering)

**O que é:** criar variáveis que contam o comportamento melhor que o dado
bruto: **razões** (parcela/renda), **tendências** (uso atual × média de 6
meses), **recência** (meses desde o último atraso) e **frequência**
(quantos meses em atraso).

**Na cooperativa:**
- **Tendência:** dois cooperados usam 60% do limite. Um sempre usou isso; o outro usava 10% e subiu em dois meses. A variável de tendência separa os dois, e o segundo é o de maior risco.
- **Recência:** quem atrasou há 1 mês tem mais risco que quem atrasou há 5 meses, mesmo com o mesmo "atraso máximo em 6 meses".
- **Comprometimento:** a parcela sobre a renda mostra a folga do orçamento, que o valor da parcela sozinho não mostra.

**Código 5.6 (SQL) · Variáveis de tendência, frequência e recência**

- **O que faz:** calcula, a partir da posição mensal, três variáveis comportamentais para a data de referência de 30/06/2024.

```sql
SELECT  id_cooperado,
        -- tendência: uso do limite no mês ÷ uso médio dos 6 meses (> 1 = subindo)
        MAX(CASE WHEN dt_posicao = '2024-06-30' THEN saldo_utilizado * 1.0 / NULLIF(limite, 0) END)
          / NULLIF(AVG(saldo_utilizado * 1.0 / NULLIF(limite, 0)), 0)  AS tendencia_uso_limite,
        -- frequência: em quantos dos 6 meses houve atraso
        SUM(CASE WHEN dias_atraso > 0 THEN 1 ELSE 0 END)               AS meses_com_atraso_6m,
        -- recência: meses desde o último atraso (nulo = não atrasou)
        DATEDIFF(MONTH, MAX(CASE WHEN dias_atraso > 0 THEN dt_posicao END), '2024-06-30')
                                                                       AS meses_desde_ultimo_atraso
FROM posicao_mensal
WHERE dt_posicao > '2023-12-31' AND dt_posicao <= '2024-06-30'
GROUP BY id_cooperado;
```

**Código 5.7 · Variáveis derivadas em pandas**

- **O que faz:** cria razões e flags a partir das colunas da base e mede o IV de cada uma.

```python
df["comprometimento"] = (df.uso_limite * 5_000) / df.renda_trat     # saldo usado (limite fictício de R$ 5 mil) / renda
df["aplicacao_sobre_renda"] = df.saldo_aplicacao / df.renda_trat     # quantos meses de renda tem aplicados
df["sem_atraso_6m"] = (df.atraso_max_6m == 0).astype(int)            # flag de "ficha limpa" recente

for v in ["comprometimento", "aplicacao_sobre_renda"]:
    print(f"IV {v}: {woe_iv(df, v)[1]:.3f}")
```

### 5.6 Seleção de atributos (feature selection)

**O que é:** reduzir a lista a poucas variáveis fortes, estáveis e pouco
redundantes. Os filtros mais comuns são:
- IV mínimo;
- correlação entre pares (|r| > 0,7, fica a de maior IV);
- estabilidade ao longo das safras;
- sentido de negócio: o sinal do coeficiente faz sentido?

**Na cooperativa:**
- **Redundância:** `saldo_aplicacao` e `aplicacao_sobre_renda` contam quase a mesma história. Fica a de maior IV, e o modelo fica mais simples de explicar ao comitê.
- **Sinal invertido:** se o modelo diz que mais tempo de casa *aumenta* o risco, a variável sai ou é investigada, mesmo com IV alto.

**Código 5.8 · Seleção por IV e correlação**

- **O que faz:** ordena as candidatas por IV, descarta as com IV abaixo de 0,02 e, entre duas variáveis muito correlacionadas, mantém a de maior IV.
- **Saída:** a lista `selecionadas`.

```python
candidatas = ["uso_limite", "atraso_max_6m", "meses_associado", "qtd_produtos",
              "saldo_aplicacao", "aplicacao_sobre_renda", "comprometimento", "idade"]

# 1) IV de cada candidata, da mais forte para a mais fraca
iv_cand = pd.Series({v: woe_iv(df, v)[1] for v in candidatas}).sort_values(ascending=False)

# 2) Correlação de Spearman (pega relações não lineares e é robusta a outliers)
corr = df[candidatas].corr(method="spearman").abs()

# 3) Entra quem tem IV ≥ 0,02 e não é muito correlacionada com quem já entrou
selecionadas = []
for v in iv_cand[iv_cand >= 0.02].index:
    if all(corr.loc[v, s] < 0.7 for s in selecionadas):
        selecionadas.append(v)
print("selecionadas:", selecionadas)
```

### 5.7 Testes de consistência e qualidade

**O que é:** verificações automáticas antes de treinar ou escorar:
- chave única;
- domínios válidos;
- percentual de nulos dentro do esperado;
- datas coerentes;
- volume compatível com o mês anterior.

Uma falha interrompe o processo antes de virar decisão errada.

**Na cooperativa:**
- **Escala trocada:** uma mudança no sistema de origem passou a gravar `uso_limite` em percentual (60) em vez de fração (0,60). O teste de domínio barra a base antes de o score de todos os cooperados despencar.
- **Base incompleta:** a carga do mês chegou com 40% menos cooperados que o mês anterior. O teste de volume impede escorar uma base parcial.

**Código 5.9 · Checagem de qualidade antes de escorar**

- **O que faz:** roda testes de chave, domínio, nulos e volume e devolve a lista de problemas, ou `["ok"]`.
- **Uso:** chamar antes de treinar e antes de cada escoragem mensal.

```python
def checa_qualidade(base, volume_mes_anterior=None):
    """Lista os problemas de qualidade encontrados na base (ou ['ok'])."""
    erros = []
    if base.id_cooperado.duplicated().any():
        erros.append("id_cooperado duplicado")
    if not base.uso_limite.between(0, 1).all():
        erros.append("uso_limite fora de [0, 1]")
    if (base.idade < 18).any():
        erros.append("idade abaixo de 18")
    if base.renda.isna().mean() > 0.10:
        erros.append(f"renda nula acima de 10% ({base.renda.isna().mean():.1%})")
    if volume_mes_anterior and abs(len(base) / volume_mes_anterior - 1) > 0.20:
        erros.append(f"volume variou mais de 20% ({len(base):,} vs {volume_mes_anterior:,})")
    return erros or ["ok"]

print(checa_qualidade(df, volume_mes_anterior=19_500))      # base normal

base_com_erro = df.assign(uso_limite=df.uso_limite * 100)   # simula a escala trocada
print(checa_qualidade(base_com_erro.head(12_000), volume_mes_anterior=19_500))
```

---

## 6. Métricas de desempenho e estabilidade

Há três famílias de métricas:
- **discriminação:** o modelo separa bons de maus? (KS, ROC/AUC, Gini)
- **classificação:** com um corte, quantos acertos e erros? (matriz de confusão e derivadas)
- **estabilidade e aderência:** a população mudou? A probabilidade prevista bate com a realizada? (PSI, calibração)

### 6.1 Matriz de confusão

**O que é:** cruza o previsto com o realizado, depois de escolher um corte.
Considere "positivo" = mau.

|  | Realizado mau | Realizado bom |
|---|---|---|
| **Previsto mau** | VP (verdadeiro positivo) | FP (bom recusado) |
| **Previsto bom** | FN (mau aprovado) | VN (verdadeiro negativo) |

**Na cooperativa:**
- **Os dois erros:** cada FN é um crédito que vira perda. Cada FP é um bom cooperado que teve o crédito negado e pode buscar outra instituição.
- **Custo em reais:** se um FN custa R$ 6.000 de perda e um FP custa R$ 900 de margem perdida, a matriz vira uma conta de resultado. O corte define o equilíbrio entre os dois erros.

**Código 6.1 · Matriz de confusão e custo dos erros**

- **O que faz:** aplica um corte de PD de 15%, monta a matriz e calcula o custo dos dois tipos de erro.

```python
from sklearn.metrics import confusion_matrix

corte_pd = 0.15                                        # PD ≥ 15% → previsto mau (nega)
teste["previsto_mau"] = (teste.pd_mau >= corte_pd).astype(int)
vn, fp, fn, vp = confusion_matrix(teste.mau, teste.previsto_mau).ravel()
print(f"VP={vp}  FP={fp}  FN={fn}  VN={vn}")

# custo: maus aprovados viram perda; bons recusados são margem que deixou de entrar
print(f"perda com maus aprovados: R$ {fn * 6_000:,.0f}   margem perdida com bons recusados: R$ {fp * 900:,.0f}")
```

### 6.2 Precision, recall, sensibilidade, especificidade e F1

**O que é:** as razões da matriz.

| Métrica | Fórmula | Pergunta |
|---|---|---|
| **Precision** (precisão) | VP / (VP + FP) | Dos que o modelo apontou como maus, quantos eram? |
| **Recall = Sensibilidade** | VP / (VP + FN) | Dos maus reais, quantos o modelo pegou? |
| **Especificidade** | VN / (VN + FP) | Dos bons reais, quantos o modelo deixou passar? |
| **F1-Score** | 2 × P × R / (P + R) | Equilíbrio entre precisão e recall |
| **Acurácia** | (VP + VN) / total | Pouco útil quando os maus são raros |

**Na cooperativa:**
- **Acurácia enganosa:** com 11% de maus, um modelo que diz "todos são bons" tem 89% de acurácia e não serve para nada. Por isso olham-se recall (quanta perda se evita) e especificidade (quantos bons se aprovam).
- **Cobrança:** no collection score, a pergunta muda. A precisão diz quantos dos que receberam ligação realmente precisavam dela.

**Código 6.2 · Métricas de classificação em um corte**

```python
from sklearn.metrics import precision_score, recall_score, f1_score

print(f"precision      {precision_score(teste.mau, teste.previsto_mau):.3f}")
print(f"recall/sensib. {recall_score(teste.mau, teste.previsto_mau):.3f}")
print(f"especificidade {vn / (vn + fp):.3f}")
print(f"F1             {f1_score(teste.mau, teste.previsto_mau):.3f}")
```

**Código 6.3 · Como as métricas mudam com o corte**

- **O que faz:** recalcula recall, especificidade e taxa de aprovação para vários cortes de PD. Mostra que não existe corte "certo" sem um objetivo de negócio.

```python
linhas = []
for c in [0.05, 0.10, 0.15, 0.20, 0.30]:
    prev = (teste.pd_mau >= c).astype(int)
    vn_, fp_, fn_, vp_ = confusion_matrix(teste.mau, prev).ravel()
    linhas.append({"corte_pd": c, "recall": vp_ / (vp_ + fn_),
                   "especificidade": vn_ / (vn_ + fp_), "aprovacao": (prev == 0).mean()})
print(pd.DataFrame(linhas).round(3).to_string(index=False))
```

### 6.3 Curva ROC, AUC e Gini

**O que é:** a curva ROC mostra sensibilidade × (1 − especificidade) para
**todos os cortes**. A **AUC** é a área sob ela: a probabilidade de um mau
sorteado ter risco maior que um bom sorteado. 0,5 é sorteio e 1,0 é perfeito.
O **Gini** é `2 × AUC − 1` e é a forma mais comum de reportar em crédito.

| Gini | Leitura usual em crédito |
|---|---|
| < 0,30 | Fraco |
| 0,30 – 0,50 | Razoável, comum em aplicação |
| 0,50 – 0,70 | Bom, comum em behavior |
| > 0,70 | Muito alto: conferir vazamento |

**Na cooperativa:** o comitê compara o Gini do modelo novo com o do atual na
mesma safra de teste. Esse é o primeiro número da apresentação. Um ganho de 5
pontos de Gini costuma justificar a troca.

**Código 6.4 · AUC e Gini**

```python
from sklearn.metrics import roc_curve

auc = roc_auc_score(teste.mau, teste.pd_mau)
print(f"AUC = {auc:.3f}   Gini = {2 * auc - 1:.3f}")
```

### 6.4 KS (Kolmogorov-Smirnov)

**O que é:** a **maior distância** entre a distribuição acumulada do score
dos bons e a dos maus. Quanto maior, melhor o modelo separa os dois grupos.
O ponto onde o KS acontece é um candidato natural a corte.

| KS | Leitura usual |
|---|---|
| < 20 | Fraco |
| 20 – 30 | Aceitável |
| 30 – 50 | Bom |
| > 60 | Desconfiar: vazamento? |

**Na cooperativa:** o behavior score tem KS de 42. No relatório mensal, uma
queda para 30 é sinal de alerta.

**Código 6.5 · KS e o corte onde ele acontece**

- **O que faz:** usa a curva ROC. O KS é a maior diferença entre a taxa de maus capturados (`tpr`) e a de bons recusados (`fpr`).

```python
fpr, tpr, cortes = roc_curve(teste.mau, teste.pd_mau)
ks = np.max(tpr - fpr)
corte_ks = cortes[np.argmax(tpr - fpr)]
print(f"KS = {ks * 100:.1f}  (no corte de PD {corte_ks:.3f})")
```

**Código 6.6 · Tabela de ordenação por decil (gains table)**

- **O que faz:** monta a tabela que mais aparece em relatórios de validação. Ela mostra, por decil de score, a taxa de maus, o percentual acumulado de maus e de bons e o KS em cada linha.
- **Leitura:** a taxa de maus deve cair do decil 1 (pior score) ao 10 (melhor), sem inversões.

```python
teste["decil_score"] = pd.qcut(teste.score.rank(method="first"), 10, labels=range(1, 11))
g = teste.groupby("decil_score", observed=True).agg(
    score_min=("score", "min"), score_max=("score", "max"),
    cooperados=("mau", "size"), maus=("mau", "sum"))
g["taxa_maus"] = g.maus / g.cooperados
g["maus_acum"] = g.maus.cumsum() / g.maus.sum()
g["bons_acum"] = (g.cooperados - g.maus).cumsum() / (g.cooperados - g.maus).sum()
g["ks"] = (g.maus_acum - g.bons_acum) * 100
print(g.round(3).to_string())
```

### 6.5 PSI (Population Stability Index)

**O que é:** mede quanto a **distribuição** de uma variável ou do score
mudou entre a base de desenvolvimento e a atual. A fórmula é
`Σ (%atual − %esperado) × ln(%atual / %esperado)`, somada por faixa. Aplicado
a cada variável de entrada, costuma ser chamado de **CSI**
(*Characteristic Stability Index*).

| PSI | Leitura |
|---|---|
| < 0,10 | Estável |
| 0,10 – 0,25 | Atenção: investigar |
| > 0,25 | Mudança relevante: avaliar recalibração |

**Na cooperativa:**
- **Campanha de consignado:** uma campanha trouxe muitos cooperados novos. O PSI do score sobe, porque a população escorada não é mais a mesma que a do desenvolvimento.
- **Mês de 13º:** em dezembro, o uso do limite cai em toda a base. Um PSI moderado nessa época pode ser sazonal, por isso vale comparar com dezembro do ano anterior.

**Código 6.7 · Função de PSI e dois cenários**

- **O que faz:** calcula o PSI com faixas definidas pelos decis da base esperada. Compara uma população igual à do desenvolvimento com outra que piorou.

```python
def psi(esperado, atual, faixas=10):
    """PSI entre duas amostras; as faixas vêm dos quantis da amostra esperada."""
    cortes = np.unique(np.quantile(esperado, np.linspace(0, 1, faixas + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf                  # abre as pontas
    e = np.histogram(esperado, cortes)[0] / len(esperado)
    a = np.histogram(atual, cortes)[0] / len(atual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)    # evita log(0)
    return np.sum((a - e) * np.log(a / e))

score_desenv = para_score(modelo.predict_proba(treino[variaveis])[:, 1])
score_hoje_igual = teste.score
score_hoje_mudou = teste.score - rng.normal(8, 4, len(teste))     # população piorou

print(f"PSI mesma população : {psi(score_desenv, score_hoje_igual):.3f}")
print(f"PSI população nova  : {psi(score_desenv, score_hoje_mudou):.3f}")
```

### 6.6 Aderência (calibração)

**O que é:** confere se a **probabilidade prevista bate com a realizada**,
faixa a faixa. Um modelo pode ordenar bem (KS alto) e mesmo assim errar o
nível. Isso pesa quando a PD alimenta preço ou provisão.

**Na cooperativa:** a faixa com PD prevista de 5% teve 9% de maus
realizados. O modelo ordena, mas subestima o risco. A provisão e a taxa
calculadas com essa PD ficariam baixas, por isso é preciso recalibrar.

**Código 6.8 · PD prevista × taxa realizada por decil**

```python
teste["decil_pd"] = pd.qcut(teste.pd_mau, 10, labels=False)
aderencia = teste.groupby("decil_pd").agg(pd_prevista=("pd_mau", "mean"), taxa_real=("mau", "mean"))
aderencia["diferenca_pp"] = (aderencia.taxa_real - aderencia.pd_prevista) * 100
print(aderencia.round(3))
```

---

## 7. Pontos de corte, faixas de risco e regras de decisão

### 7.1 Ponto de corte (cut-off) e threshold

**O que é:** o score a partir do qual a decisão muda: aprovar ou negar,
cobrar por SMS ou por ligação. Escolhe-se o corte pelo **objetivo de
negócio**, não pela estatística: taxa de aprovação desejada, inadimplência
máxima aceita ou resultado financeiro esperado.

**Na cooperativa:**
- **Limite de inadimplência:** a diretoria aceita no máximo 4% de inadimplência na carteira aprovada. A tabela abaixo mostra o maior volume aprovado que respeita esse limite.
- **Resultado:** com margem de R$ 900 por bom e perda de R$ 6.000 por mau, a tabela também mostra o corte que maximiza o resultado. Os dois cortes raramente coincidem, e escolher entre eles é uma decisão de apetite a risco.

**Código 7.1 · Tabela de cortes: aprovação, inadimplência e resultado**

- **O que faz:** simula cortes de 480 a 700 pontos e calcula, para cada um, a taxa de aprovação, a inadimplência dos aprovados e o resultado financeiro.

```python
MARGEM_BOM, PERDA_MAU = 900, 6_000

linhas = []
for corte in range(480, 701, 20):
    aprovados = teste[teste.score >= corte]
    maus = aprovados.mau.sum()
    linhas.append({
        "corte": corte,
        "aprovacao": len(aprovados) / len(teste),
        "inad_aprovados": aprovados.mau.mean() if len(aprovados) else np.nan,
        "resultado_R$": (len(aprovados) - maus) * MARGEM_BOM - maus * PERDA_MAU,
    })
tabela_corte = pd.DataFrame(linhas)
print(tabela_corte.round(3).to_string(index=False))

dentro = tabela_corte[tabela_corte.inad_aprovados <= 0.04]
print("corte para inadimplência ≤ 4%:", dentro.corte.min())
print("corte de maior resultado    :", tabela_corte.loc[tabela_corte["resultado_R$"].idxmax(), "corte"])
```

### 7.2 Faixas de risco (rating)

**O que é:** agrupar o score em poucas faixas (A a E) com **inadimplência
crescente e bem separada**. As faixas são a linguagem comum entre o modelo
e a política: "faixa A tem renovação automática", "faixa E não recebe oferta".

**Na cooperativa:**
- **Linguagem do gerente:** o gerente não vê "score 612". Vê "Rating B", com o limite e a taxa que a política define para essa faixa.
- **Preço:** a taxa de juros pode variar por rating, cobrando de cada faixa o risco que ela traz.

**Código 7.2 · Criar o rating A–E e conferir a monotonicidade**

- **O que faz:** divide o score em cinco faixas de mesmo tamanho e verifica se a inadimplência cai de E para A sem inversões.

```python
teste["rating"] = pd.qcut(teste.score, 5, labels=["E", "D", "C", "B", "A"])
faixas = teste.groupby("rating", observed=True).agg(
    cooperados=("score", "size"), score_min=("score", "min"),
    score_max=("score", "max"), inadimplencia=("mau", "mean"))
print(faixas.sort_index(ascending=False).round(3))

taxas = faixas.sort_index(ascending=False).inadimplencia      # de A para E
print("monotônico:", taxas.is_monotonic_increasing)
```

**Código 7.3 · Taxa de juros por rating (precificação pelo risco)**

- **O que faz:** calcula uma taxa mensal mínima por rating que cobre o custo de captação, a perda esperada e a margem. A fórmula é simplificada, só para ilustrar a lógica.

```python
CUSTO_CAPTACAO, MARGEM, LGD = 0.009, 0.006, 0.60            # ao mês; LGD = perda quando há default

preco = faixas[["inadimplencia"]].copy()
preco["perda_esperada_mes"] = preco.inadimplencia * LGD / 12   # PD anual → aproximação mensal
preco["taxa_minima_mes"] = CUSTO_CAPTACAO + MARGEM + preco.perda_esperada_mes
print((preco * 100).round(2).sort_index(ascending=False))     # em %
```

### 7.3 Regras de decisão

**O que é:** a política completa combina **filtros** (regras eliminatórias),
o **score** e as **alçadas**. A ordem importa e precisa estar documentada.

**Na cooperativa:**

```text
1. Filtros:  restrição grave no birô ou prejuízo anterior na cooperativa → NEGA
2. Score:    rating D ou E                                → NEGA
             rating C                                     → ANALISA (gerente)
             rating A ou B                                → segue
3. Limite:   valor ≤ 30% da renda e ≤ teto da faixa       → APROVA automático
             acima disso                                  → ALÇADA do comitê
```

**Código 7.4 · Motor de decisão simples**

- **O que faz:** aplica filtro, rating e alçada a cada cooperado, na ordem da política, e resume quantos caem em cada decisão e a inadimplência de cada grupo.

```python
def decide(c):
    """Aplica a política na ordem: filtro → score → alçada."""
    if c.atraso_max_6m >= 60:                 # filtro eliminatório
        return "Nega (filtro)"
    if c.rating in ("D", "E"):
        return "Nega (score)"
    if c.rating == "C":
        return "Análise do gerente"
    return "Aprova automático"

teste["decisao"] = teste.apply(decide, axis=1)
print(teste.groupby("decisao").agg(qtd=("mau", "size"), inad=("mau", "mean")).round(3))
```

---

## 8. Análise de carteira: vintage, MOB, rolagem e curvas de inadimplência

**O que é:**
- **Safra (vintage):** os contratos concedidos num mesmo mês.
- **MOB (*months on book*):** quantos meses se passaram desde a concessão.
- **Curva de vintage:** a inadimplência acumulada (por exemplo, % que atingiu 90+ dias) por MOB, uma linha por safra.

Comparar safras no **mesmo MOB** mostra se a qualidade da concessão melhorou
ou piorou, sem a distorção da idade dos contratos.

**Na cooperativa:**
- **Safra pior:** a carteira total tem 3% de inadimplência e parece tranquila. Mas a curva mostra que a safra de out/2024, no MOB 6, já tem o dobro de maus da safra de jan/2024 no mesmo MOB. A política afrouxada em setembro está piorando a concessão, e a carteira total só mostraria isso meses depois.
- **Produto novo:** a primeira safra do crédito para energia solar é acompanhada mês a mês contra a curva do crédito pessoal, para saber cedo se o produto se comporta como o esperado.

**Código 8.1 (SQL) · Curva de vintage por safra e MOB**

- **O que faz:** para cada safra de concessão e cada MOB, calcula o percentual de contratos que já atingiram 90+ dias de atraso.
- **Premissa:** a coluna `dias_atraso_max_ate_aqui` guarda o maior atraso do contrato até a data da posição.

```sql
SELECT  FORMAT(ct.dt_concessao, 'yyyy-MM')                          AS safra,
        DATEDIFF(MONTH, ct.dt_concessao, p.dt_posicao)              AS mob,
        COUNT(DISTINCT ct.id_contrato)                               AS contratos,
        COUNT(DISTINCT CASE WHEN p.dias_atraso_max_ate_aqui > 90
                            THEN ct.id_contrato END) * 1.0
          / COUNT(DISTINCT ct.id_contrato)                           AS pct_90_mais
FROM contrato AS ct
JOIN posicao_mensal AS p ON p.id_contrato = ct.id_contrato
GROUP BY FORMAT(ct.dt_concessao, 'yyyy-MM'), DATEDIFF(MONTH, ct.dt_concessao, p.dt_posicao)
ORDER BY safra, mob;
```

**Código 8.2 · Curva de vintage em pandas**

- **O que faz:** gera 4 safras fictícias com risco crescente e monta a tabela MOB × safra.
- **Leitura:** a safra só mostra os meses que já viveu até a data de corte (jun/2025). O resto fica vazio.

```python
# 1) Contratos fictícios: cada safra com um risco mensal diferente
safras = {"2024-01": .010, "2024-04": .012, "2024-07": .016, "2024-10": .022}
data_corte = pd.Period("2025-06", freq="M")
contratos = pd.DataFrame({"safra": rng.choice(list(safras), 4_000)})
risco = contratos.safra.map(safras)
# MOB em que o contrato atingiu 90+ (999 = nunca atingiu)
contratos["mob_default"] = np.where(rng.random(len(contratos)) < risco * 12,
                                    rng.integers(1, 13, len(contratos)), 999)

# 2) % acumulado de 90+ por MOB, só até o MOB já observado
vintage = {}
for safra, grupo in contratos.groupby("safra"):
    mob_max = (data_corte - pd.Period(safra, freq="M")).n
    vintage[safra] = {mob: (grupo.mob_default <= mob).mean() if mob <= mob_max else np.nan
                      for mob in range(1, 13)}
print((pd.DataFrame(vintage) * 100).round(1))     # linhas = MOB, colunas = safra
```

**Curvas de inadimplência e rolagem (roll rate):** acompanham quanto da
carteira passa de uma faixa de atraso para a seguinte a cada mês (em dia →
1–30 → 31–60 → 61–90 → 90+). Se a rolagem de 1–30 para 31–60 sobe, a
cobrança precoce está perdendo eficácia antes de a inadimplência aparecer.

**Na cooperativa:** a matriz de rolagem mostra que 25% de quem estava em
1–30 dias passou para 31–60. No trimestre anterior eram 18%. A cobrança
reforça a ligação no 10º dia de atraso.

**Código 8.3 · Matriz de rolagem entre faixas de atraso**

- **O que faz:** simula a faixa de atraso de 10 mil contratos em dois meses seguidos e calcula o percentual que fica, melhora ou piora.
- **Leitura:** cada linha é a faixa do mês anterior e soma 100%. A diagonal é quem ficou na mesma faixa, e à direita dela está quem piorou.

```python
faixas_atraso = ["em dia", "1-30", "31-60", "61-90", "90+"]
# probabilidades de transição fictícias (linha = de, coluna = para)
transicao = np.array([[.95, .05, .00, .00, .00],
                      [.55, .20, .25, .00, .00],
                      [.25, .10, .20, .45, .00],
                      [.10, .05, .05, .20, .60],
                      [.05, .00, .00, .00, .95]])

rng_roll = np.random.default_rng(3)
mes_ant = rng_roll.choice(5, 10_000, p=[.85, .07, .04, .02, .02])
mes_atual = np.array([rng_roll.choice(5, p=transicao[i]) for i in mes_ant])

rolagem = pd.crosstab(pd.Categorical.from_codes(mes_ant, faixas_atraso),
                      pd.Categorical.from_codes(mes_atual, faixas_atraso),
                      rownames=["mês anterior"], colnames=["mês atual"], normalize="index")
print((rolagem * 100).round(1))
```

---

## 9. Segmentação de cooperados

**O que é:** dividir a base em grupos **homogêneos por dentro e diferentes
entre si**.
- **Segmentação estatística:** clustering (k-means) ou árvores de decisão. A árvore separa grupos de risco com regras legíveis.
- **Segmentação comportamental:** padrões de uso, como tomador, aplicador, transacional ou inativo.

Também é comum segmentar **antes** de modelar: um modelo para pessoa física
e outro para produtor rural.

**Na cooperativa:** o k-means separa quatro perfis, e cada um recebe uma
oferta e uma régua de cobrança diferentes.
- **Aplicadores conservadores:** alto saldo aplicado e pouco crédito. Recebem ofertas de investimento e consórcio.
- **Tomadores intensos:** limite alto e muito usado. Recebem acompanhamento de risco e portabilidade de dívida para uma linha mais barata.
- **Jovens em construção:** pouco tempo de casa e poucos produtos. Recebem educação financeira e cartão de entrada.
- **Relacionamento pleno:** muitos produtos. Têm prioridade de atendimento e retenção.

**Código 9.1 · Segmentação comportamental com k-means**

- **O que faz:** padroniza cinco variáveis, agrupa os cooperados em 4 segmentos e mostra o perfil médio de cada um.
- **Depois:** dê um nome a cada segmento lendo o perfil. O algoritmo só devolve números.

```python
from sklearn.cluster import KMeans

vars_seg = ["uso_limite", "saldo_aplicacao_w", "qtd_produtos", "meses_associado", "idade"]

# 1) Padroniza: sem isso, saldo (em milhares) dominaria uso_limite (0 a 1)
X_seg = StandardScaler().fit_transform(df[vars_seg])

# 2) Agrupa em 4 segmentos
df["segmento"] = KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(X_seg)

# 3) Perfil médio, taxa de maus e tamanho de cada segmento
perfil = df.groupby("segmento")[vars_seg + ["mau"]].mean()
perfil["tamanho"] = df.segmento.value_counts(normalize=True)
print(perfil.round(2).to_string())
```

**Código 9.2 · Segmentos de risco com árvore de decisão**

- **O que faz:** treina uma árvore rasa, com folhas de no mínimo mil cooperados, e mostra as regras e a taxa de maus de cada folha.
- **Uso:** cada folha vira uma regra de política legível, por exemplo "atraso ≤ 22 dias e uso ≤ 38%".

```python
from sklearn.tree import DecisionTreeClassifier, export_text

vars_arv = ["uso_limite", "atraso_max_6m", "meses_associado"]
arvore = DecisionTreeClassifier(max_depth=2, min_samples_leaf=1_000, random_state=42)
arvore.fit(df[vars_arv], df.mau)
print(export_text(arvore, feature_names=vars_arv))

# a árvore marca todas as folhas como "classe 0" porque os maus são minoria;
# o que interessa é a TAXA de maus de cada folha
df["folha"] = arvore.apply(df[vars_arv])
print(df.groupby("folha").mau.agg(cooperados="size", taxa_maus="mean").round(3))
```

---

## 10. Políticas, Champion-Challenger e estudos de impacto

### 10.1 Champion-Challenger

**O que é:** um teste controlado entre duas estratégias.
- A atual (**champion**) segue na maior parte da base.
- A alternativa (**challenger**) roda numa amostra **aleatória** pequena, por exemplo 10%.

Depois de um período de maturação, comparam-se os resultados. O sorteio
garante que a diferença vem da estratégia, e não do perfil dos cooperados.

**Na cooperativa:**
- **Cobrança:** o champion liga a partir do 5º dia de atraso. O challenger manda WhatsApp no 1º dia e liga só no 10º. Depois de 3 meses, compara-se a taxa de cura e o custo por real recuperado.
- **Modelo novo:** o behavior v3 decide o limite de 10% da base enquanto o v2 segue nos outros 90%. Depois de 12 meses, compara-se a inadimplência e o volume concedido.

**Código 10.1 · Sortear os grupos e testar a diferença**

- **O que faz:** sorteia 10% dos cooperados em atraso para o challenger, simula o resultado e aplica um teste z de proporções para saber se a diferença é real ou acaso.

```python
# 1) Sorteio: 10% challenger, 90% champion
cobranca["grupo"] = np.where(rng.random(len(cobranca)) < 0.10, "challenger", "champion")

# 2) Resultado fictício: o desafiante cura 4 p.p. a mais
efeito = np.where(cobranca.grupo == "challenger", 0.04, 0)
cobranca["curou"] = (rng.random(len(cobranca)) < cobranca.p_cura + efeito).astype(int)

# 3) Teste z de diferença entre duas proporções
res = cobranca.groupby("grupo").curou.agg(n="size", taxa="mean")
p1, p2 = res.loc["challenger", "taxa"], res.loc["champion", "taxa"]
n1, n2 = res.loc["challenger", "n"], res.loc["champion", "n"]
p = (p1 * n1 + p2 * n2) / (n1 + n2)
z = (p1 - p2) / np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
print(res.round(3), f"\nz = {z:.2f}  (|z| > 1,96 ≈ diferença significativa a 95%)")
```

**Código 10.2 · Tamanho mínimo do grupo challenger**

- **O que faz:** estima quantos cooperados o challenger precisa ter para detectar uma melhora de 4 p.p. na taxa de cura (de 63% para 67%), com 95% de confiança e 80% de poder.
- **Por que importa:** um challenger pequeno demais termina o teste sem conclusão.

```python
z_alfa, z_beta = 1.96, 0.84             # 95% de confiança, 80% de poder
p_a, p_b = 0.63, 0.67
p_med = (p_a + p_b) / 2
n_grupo = ((z_alfa * np.sqrt(2 * p_med * (1 - p_med))
            + z_beta * np.sqrt(p_a * (1 - p_a) + p_b * (1 - p_b))) ** 2) / (p_b - p_a) ** 2
print(f"mínimo por grupo: {np.ceil(n_grupo):.0f} cooperados")
```

### 10.2 Políticas desafiantes

**O que é:** uma proposta de mudança na política de crédito, cobrança ou
negócios: novo corte, nova faixa de limite, novo filtro. Ela é **mensurada
antes** de substituir a atual, por simulação histórica (10.3) e depois por
Champion-Challenger (10.1).

**Na cooperativa:**
- **Corte por tempo de casa:** a política desafiante propõe baixar o corte de 600 para 580 pontos para cooperados com mais de 5 anos de casa, com o argumento de que o tempo de relacionamento reduz o risco. A simulação mede quantos entram e a inadimplência deles antes de ir para o piloto.
- **Cobrança antecipada:** outra desafiante propõe antecipar a renegociação para o 30º dia de atraso, em vez do 60º, para os cooperados com collection score baixo.

### 10.3 Estudo de impacto: swap-in e swap-out

**O que é:** aplicar a política nova sobre a base histórica e comparar com a
atual.
- **Swap-in:** quem passaria a ser aprovado.
- **Swap-out:** quem deixaria de ser aprovado.

O resultado líquido depende da inadimplência desses dois grupos. Como os
recusados no passado não têm desempenho observado, os swap-ins exigem
cuidado: técnicas de *reject inference* ou um piloto controlado.

**Na cooperativa:** a política desafiante da seção 10.2 é aplicada sobre a
safra de teste. A pergunta é se os cooperados que ela passa a aprovar são
melhores ou piores que os que ela deixa de aprovar.

**Código 10.3 · Swap-in, swap-out e impacto da política desafiante**

- **O que faz:** compara a política atual (score ≥ 600) com a desafiante (score ≥ 580 para quem tem 5 ou mais anos de casa, ou score ≥ 610 para os demais).
- **Saída:** a matriz de troca, a inadimplência dos swap-ins e dos swap-outs, e a aprovação e a inadimplência antes e depois.

```python
atual = teste.score >= 600
nova = ((teste.score >= 580) & (teste.meses_associado >= 60)) | (teste.score >= 610)

# 1) Matriz de troca entre as duas políticas
impacto = pd.crosstab(atual.map({True: "aprova hoje", False: "nega hoje"}),
                      nova.map({True: "aprova nova", False: "nega nova"}))
print(impacto)

# 2) Quem entra e quem sai, e a qualidade de cada grupo
swap_in = teste[~atual & nova]
swap_out = teste[atual & ~nova]
print(f"swap-in : {len(swap_in):>5}  inadimplência {swap_in.mau.mean():.1%}")
print(f"swap-out: {len(swap_out):>5}  inadimplência {swap_out.mau.mean():.1%}")

# 3) Efeito total na carteira aprovada
print(f"aprovação {atual.mean():.1%} → {nova.mean():.1%}   "
      f"inadimplência {teste[atual].mau.mean():.1%} → {teste[nova].mau.mean():.1%}")
```

Nesta base fictícia, a desafiante aprova cerca de 14 p.p. a mais. Mas os
swap-ins têm 12% de maus, contra 8% dos swap-outs, e a inadimplência da
carteira aprovada sobe de 5,4% para 7,0%. O tempo de casa não compensa o
score mais baixo. O estudo barra a mudança antes de qualquer piloto, e é
para isso que ele serve.

**Código 10.4 · Simulação de impacto financeiro de vários cortes por segmento**

- **O que faz:** testa, só para cooperados com 5 ou mais anos de casa, cortes de 560 a 620. Para cada corte calcula o resultado financeiro total, mantendo o corte de 600 para os demais.
- **Uso:** encontrar o corte que maximiza o resultado antes de levar a proposta ao comitê.

```python
veteranos = teste.meses_associado >= 60
sim = []
for corte_vet in range(560, 621, 10):
    aprova = (veteranos & (teste.score >= corte_vet)) | (~veteranos & (teste.score >= 600))
    maus = teste[aprova].mau.sum()
    sim.append({"corte_veteranos": corte_vet, "aprovacao": aprova.mean(),
                "resultado_R$": (aprova.sum() - maus) * MARGEM_BOM - maus * PERDA_MAU})
print(pd.DataFrame(sim).round(3).to_string(index=False))
```

### 10.4 Estratégias de concessão, limite, cobrança e recuperação

**O que é:** o score sozinho não decide nada. A estratégia traduz faixa de
risco e perfil em **ação**.

| Estratégia | Como o score entra | Exemplo na cooperativa |
|---|---|---|
| Concessão | Corte e alçada por rating | A e B automático, C com gerente, D e E nega |
| Manutenção de limite | Matriz rating × uso | Rating A com uso > 70%: aumento de 20% |
| Cobrança | Collection score define régua e canal | Cura alta: SMS; cura baixa: renegociação no 15º dia |
| Recuperação | Recuperação esperada define o acordo | Desconto máximo = 1 − recuperação esperada − custo |
| Relacionamento | Propensão e churn definem oferta | Alto churn com alto valor: visita do gerente |

**Código 10.5 · Matriz de limite por rating e renda**

- **O que faz:** define o multiplicador do limite atual por rating e faixa de renda e aplica a três cooperados de exemplo.

```python
multiplicador = pd.DataFrame(
    [[1.3, 1.5, 1.8], [1.1, 1.3, 1.5], [1.0, 1.0, 1.1], [0.8, 0.9, 1.0], [0.5, 0.6, 0.7]],
    index=["A", "B", "C", "D", "E"], columns=["até 3k", "3k–8k", "acima 8k"])
print(multiplicador)

exemplos = pd.DataFrame({"cooperado": ["Ana", "Bruno", "Carla"], "rating": ["A", "C", "E"],
                         "faixa_renda": ["3k–8k", "acima 8k", "até 3k"], "limite_atual": [4_000, 10_000, 2_000]})
exemplos["novo_limite"] = [multiplicador.loc[r, f] * l for r, f, l in
                           zip(exemplos.rating, exemplos.faixa_renda, exemplos.limite_atual)]
print(exemplos)
```

---

## 11. Monitoramento, backtesting e drift

### 11.1 Backtesting

**O que é:** comparar, depois da maturação, **o que o modelo previu com o
que aconteceu**, safra por safra. Confere discriminação (KS, Gini) e
aderência (PD prevista × realizada).

**Na cooperativa:** a cada trimestre, a safra escorada há 12 meses
amadurece. A PD média prevista era 6,1% e a realizada foi 6,4%, dentro da
tolerância definida na política de modelos.

**Código 11.1 · Backtesting por safra**

- **O que faz:** para cada safra, compara a PD média prevista com a taxa realizada e calcula o KS.
- **Leitura:** em safras pequenas o KS oscila. Olhe a tendência, não um mês isolado.

```python
def ks_de(y, p):
    """KS (0 a 100) entre o alvo y e a probabilidade p."""
    fpr, tpr, _ = roc_curve(y, p)
    return np.max(tpr - fpr) * 100

backtest = teste.groupby("safra").apply(
    lambda g: pd.Series({"cooperados": len(g), "pd_prevista": g.pd_mau.mean(),
                         "taxa_real": g.mau.mean(), "KS": ks_de(g.mau, g.pd_mau)}),
    include_groups=False)
print(backtest.round(3))
```

### 11.2 Data drift

**O que é:** mudança na **distribuição das variáveis de entrada**, medida
pelo PSI (CSI) de cada uma. O modelo pode continuar certo, mas está sendo
usado numa população diferente da que conheceu.

**Na cooperativa:**
- **Incorporação:** a cooperativa incorporou uma singular do interior. O CSI de `meses_associado` dispara, porque os cooperados incorporados aparecem como "novos" no sistema, apesar de anos de relacionamento na cooperativa de origem.
- **Mudança de produto:** o limite do cartão passou a ser concedido maior na abertura da conta. O `uso_limite` de toda a base cai, sem que o risco tenha mudado.

**Código 11.2 · CSI de cada variável num mês com incorporação**

- **O que faz:** simula 25% de cooperados incorporados, com tempo de casa zerado, e calcula o PSI de cada variável contra o treino.

```python
mes_atual = teste.copy()
incorporados = rng.random(len(mes_atual)) < 0.25
mes_atual.loc[incorporados, "meses_associado"] = rng.integers(1, 6, incorporados.sum())

csi = {v: psi(treino[v], mes_atual[v]) for v in variaveis}
print(pd.Series(csi).round(3).sort_values(ascending=False))
```

### 11.3 Model drift e perda de poder discriminatório

**O que é:** a **relação** entre variáveis e alvo muda, também chamado de
*concept drift*. O sintoma é o KS ou o Gini caindo safra após safra, mesmo
com PSI estável. Causas comuns: mudança de política, de produto, de cenário
econômico ou de comportamento.

**Na cooperativa:** com o Pix parcelado, o uso do cheque especial deixou de
indicar aperto financeiro como antes. O KS do behavior score caiu de 42 para
31 em seis safras.

| Sintoma | Diagnóstico provável | Ação |
|---|---|---|
| PSI alto, KS estável | População mudou, relação não | Monitorar; recalibrar se a aderência sair |
| PSI estável, KS caindo | Relação mudou (concept drift) | Reconstruir o modelo |
| KS estável, PD prevista ≠ realizada | Nível de risco mudou | **Recalibrar** (ajuste de intercepto ou escala) |
| Tudo piorando | Mudança estrutural | Reconstruir com safras recentes |

**Código 11.3 · Recalibração pelo nível de risco recente**

- **O que faz:** ajusta todas as PDs por um mesmo fator de chance, de modo que a média passe a bater com a inadimplência recente.
- **Leitura:** a ordenação não muda, então a AUC é a mesma. Só muda o nível da PD.

```python
taxa_recente = 0.14                      # inadimplência observada nas safras recentes
odds_ajuste = (taxa_recente / (1 - taxa_recente)) / (teste.pd_mau.mean() / (1 - teste.pd_mau.mean()))
odds = teste.pd_mau / (1 - teste.pd_mau) * odds_ajuste
teste["pd_recalibrada"] = odds / (1 + odds)
print(f"PD média: {teste.pd_mau.mean():.3f} → {teste.pd_recalibrada.mean():.3f}"
      f"   (AUC inalterada: {roc_auc_score(teste.mau, teste.pd_recalibrada):.3f})")
```

**Código 11.4 · Semáforo de monitoramento com gatilhos**

- **O que faz:** junta KS, PSI e diferença de aderência numa função que devolve verde, amarelo ou vermelho e a ação recomendada.
- **Uso:** rodar a cada fechamento e enviar o resultado ao comitê de modelos.

```python
def semaforo(ks_atual, ks_desenv, psi_score, pd_prevista, taxa_real):
    """Classifica a saúde do modelo pelos gatilhos da política de modelos."""
    alertas = []
    if ks_atual < 0.75 * ks_desenv:
        alertas.append(("vermelho", f"KS caiu mais de 25% ({ks_desenv:.0f} → {ks_atual:.0f}): reconstruir"))
    elif ks_atual < 0.90 * ks_desenv:
        alertas.append(("amarelo", f"KS caiu mais de 10% ({ks_desenv:.0f} → {ks_atual:.0f}): investigar"))
    if psi_score > 0.25:
        alertas.append(("vermelho", f"PSI {psi_score:.2f}: população mudou"))
    elif psi_score > 0.10:
        alertas.append(("amarelo", f"PSI {psi_score:.2f}: acompanhar"))
    if abs(taxa_real - pd_prevista) > 0.02:
        alertas.append(("amarelo", f"aderência: previsto {pd_prevista:.1%} × real {taxa_real:.1%}: recalibrar"))
    return alertas or [("verde", "modelo saudável")]

for cor, msg in semaforo(ks_atual=31, ks_desenv=ks * 100, psi_score=0.12,
                         pd_prevista=0.108, taxa_real=0.14):
    print(f"[{cor}] {msg}")
```

---

## 12. Governança de dados e de modelos

**O que é:** o conjunto de práticas que permite **explicar, reproduzir e
auditar** qualquer decisão tomada por um modelo:
- **documentação metodológica:** como foi feito e por quê;
- **versionamento:** código, dados e modelo;
- **rastreabilidade:** qual versão escorou qual cooperado;
- **reprodutibilidade:** rodar de novo dá o mesmo resultado;
- **gestão da qualidade dos dados.**

No Brasil, a gestão de risco de crédito segue as normas do CMN e do Banco
Central. É o caso da Resolução CMN nº 4.966/2021, que trata de perda
esperada. Na prática, isso exige modelos documentados e validados.

**Na cooperativa:** a auditoria pergunta por que um cooperado teve o limite
reduzido em março. A resposta precisa vir em minutos:
- a versão do modelo (`behavior_pf v2.3`);
- a base usada (hash e data);
- o score calculado (548, rating D);
- a regra de política aplicada.

**Código 12.1 · Registro (metadados) de uma versão do modelo**

- **O que faz:** gera a "ficha" da versão treinada: nome, versão, data, impressão digital (hash) da base de treino, safras, variáveis, semente e métricas.
- **Uso:** guardar junto do modelo (no git, no MLflow ou em ferramenta equivalente).

```python
import hashlib, json
from datetime import datetime

registro = {
    "modelo": "behavior_pf",
    "versao": "2.3.0",
    "treinado_em": datetime.now().isoformat(timespec="seconds"),
    # hash: se a base mudar uma vírgula, o hash muda → prova de qual base foi usada
    "base_hash": hashlib.sha256(pd.util.hash_pandas_object(treino[variaveis + ["mau"]]).values).hexdigest()[:16],
    "safras_treino": sorted(treino.safra.unique().tolist()),
    "variaveis": variaveis,
    "semente": 42,
    "metricas_teste": {"KS": round(ks * 100, 1), "Gini": round(2 * auc - 1, 3)},
}
print(json.dumps(registro, indent=2, ensure_ascii=False))
```

**Código 12.2 · Histórico de escoragem rastreável**

- **O que faz:** grava cada score com a data, a versão do modelo, o rating e a decisão. É a tabela que responde à auditoria.

```python
score_historico = teste[["id_cooperado", "score", "rating", "decisao"]].assign(
    dt_referencia="2025-03-31", versao_modelo=registro["versao"])

# resposta à auditoria: por que o cooperado X teve essa decisão?
coop = score_historico.iloc[0].id_cooperado
print(score_historico[score_historico.id_cooperado == coop].to_string(index=False))
```

**Modelo de documentação (model card):**

```markdown
# behavior_pf v2.3.0
- Objetivo e decisão apoiada:  renovação de limite PF
- Público e exclusões:          PF com 6+ meses de conta; exclui prejuízo e falecidos
- Definição de mau:             atraso > 90 dias em 12 meses
- Janelas e safras:             obs. 6m, desempenho 12m; treino jan–set/24, OOT out–dez/24
- Variáveis (IV, sinal):        uso_limite (0,19, +) · atraso_max_6m (0,17, +) · ...
- Técnica:                      regressão logística; escala 600 pts = 10:1, PDO 20
- Desempenho:                   KS 32 · Gini 0,45 · PSI treino×OOT 0,00
- Limitações conhecidas:        cooperados incorporados sem histórico
- Monitoramento:                mensal; gatilhos KS < 75% do desenvolvimento ou PSI > 0,25
- Aprovações:                   desenvolvimento · validação independente · comitê de risco
```

**Qualidade e enriquecimento dos dados:** a governança inclui manter um
**dicionário de dados**, regras de qualidade automatizadas (Código 5.9),
registro de inconsistências com responsável e prazo, e um mapa de lacunas
que viram oportunidades. Exemplos de lacunas: renda desatualizada há mais
de 2 anos, falta do dado de Open Finance, ausência de dados de produção
rural.

**Código 12.3 · Mapa de qualidade por coluna**

- **O que faz:** gera, para cada coluna, o tipo, o percentual de nulos, a quantidade de valores distintos e um exemplo. É o ponto de partida do dicionário de dados.

```python
colunas = ["id_cooperado", "safra", "idade", "renda", "meses_associado",
           "qtd_produtos", "uso_limite", "atraso_max_6m", "saldo_aplicacao", "regiao"]
mapa = pd.DataFrame({
    "tipo": df[colunas].dtypes.astype(str),
    "pct_nulos": df[colunas].isna().mean().round(3),
    "distintos": df[colunas].nunique(),
    "exemplo": df[colunas].iloc[0],
})
print(mapa.to_string())
```

---

## 13. Indicadores, dashboards e comunicação executiva

### 13.1 Indicadores e dashboards

**O que é:** o painel de acompanhamento junta, por safra e por mês,
desempenho, estabilidade e resultado de negócio. É ele que dispara a
conversa de recalibração.

| Bloco | Indicadores |
|---|---|
| Modelo | KS, Gini, PSI do score, CSI das variáveis, PD prevista × realizada |
| Política | Taxa de aprovação, % por rating, % de decisões manuais (overrides) |
| Carteira | Inadimplência 15+/90+, vintage por MOB, rolagem entre faixas |
| Cobrança | Taxa de cura por faixa do collection score, R$ recuperado, custo por R$ |
| Negócio | Volume concedido, receita, perda esperada, resultado por faixa |

**Na cooperativa:** a tabela mensal abaixo alimenta o painel no Power BI. O
modelo grava cada score com a versão, e o painel cruza com o desempenho
quando ele amadurece.

**Código 13.1 (SQL) · Base do painel de monitoramento**

- **O que faz:** agrega o histórico de escoragem por mês, versão e rating, com a PD prevista, a taxa realizada (quando já maturou) e a taxa de aprovação.

```sql
SELECT  s.dt_referencia,
        s.versao_modelo,
        s.rating,
        COUNT(*)                                       AS cooperados,
        AVG(s.pd_prevista)                             AS pd_prevista,
        AVG(CASE WHEN d.mau = 1 THEN 1.0 ELSE 0 END)   AS taxa_realizada,   -- nula até maturar
        SUM(CASE WHEN s.decisao = 'APROVA' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS taxa_aprovacao
FROM score_historico AS s
LEFT JOIN desempenho_12m AS d
       ON d.id_cooperado = s.id_cooperado AND d.dt_referencia = s.dt_referencia
GROUP BY s.dt_referencia, s.versao_modelo, s.rating;
```

**Código 13.2 · Resumo mensal automático para a gestão**

- **O que faz:** reúne os números principais do mês num texto curto, pronto para e-mail ou para a capa do painel.

```python
aprov = (teste.decisao == "Aprova automático").mean()
print(f"""Behavior score · fechamento de março
- Cooperados escorados: {len(teste):,}
- KS {ks * 100:.0f} · Gini {2 * auc - 1:.2f} · PSI do score {psi(score_desenv, teste.score):.2f}
- Aprovação automática: {aprov:.1%}
- Inadimplência por rating: """ + " · ".join(f"{r} {t:.1%}" for r, t in
                                              faixas.inadimplencia.sort_index(ascending=False).items()))
```

### 13.2 Recomendação executiva

**O que é:** traduzir a análise em **decisão**. A alta gestão precisa de
três coisas, nesta ordem:
1. **o que fazer;**
2. **o que se ganha e o que se arrisca, em reais;**
3. **como saber se deu certo.**

A estatística vai para o anexo.

**Na cooperativa:**

- **Exemplo 1, recomendação de mudança** (valores ilustrativos):

  > **Recomendação:** baixar o corte de aprovação de 600 para 580 pontos para
  > cooperados com 5 anos ou mais de relacionamento.
  >
  > **Impacto esperado:** a aprovação sobe 4 p.p. (cerca de 900 cooperados a
  > mais por ano) e a inadimplência da carteira aprovada cai de 3,9% para 3,7%.
  > O resultado líquido estimado é de R$ 610 mil a mais por ano.
  >
  > **Risco e controle:** os novos aprovados não têm histórico de desempenho.
  > Proposta: piloto com 10% da base por 6 meses (Champion-Challenger) e
  > suspensão automática se a inadimplência do grupo passar de 5%.

- **Exemplo 2, recomendação de não mudar**, com base no Código 10.3:

  > **Recomendação:** manter o corte atual de 600 pontos.
  >
  > **Motivo:** a proposta de reduzir o corte para veteranos aprovaria 14 p.p.
  > a mais, mas os novos aprovados teriam 12% de inadimplência. A carteira
  > aprovada iria de 5,4% para 7,0%.
  >
  > **Próximo passo:** testar o corte de 590 apenas para cooperados com
  > aplicação acima de R$ 10 mil, perfil em que a simulação indica menor risco.

---

## Índice dos códigos

| Código | Tema | Linguagem |
|---|---|---|
| 0.1 | Gerar a carteira fictícia | Python |
| 2.1 | Credit score de aplicação e decisão de proposta | Python |
| 2.2 | Behavior score em pontos | Python |
| 2.3 | Pesos do modelo | Python |
| 2.4 | Collection score e régua | Python |
| 2.5 | Custo da régua de cobrança | Python |
| 2.6 | Propensão e lift por decil | Python |
| 2.7 | Retorno da campanha | Python |
| 2.8 | Churn e lista priorizada | Python |
| 2.9 | Lembrete preventivo | Python |
| 2.10 | Recuperação e propostas de acordo | Python |
| 3.1 | Logística × gradient boosting | Python |
| 4.1 | Base analítica por data de referência | SQL |
| 4.2 | Validação fora do tempo | Python |
| 5.1 | Raio-X da base | Python |
| 5.2 | Missing: flag e imputação | Python |
| 5.3 | Outliers: winsorização e IQR | Python |
| 5.4 – 5.5 | WoE, IV e ranking | Python |
| 5.6 | Tendência, frequência, recência | SQL |
| 5.7 | Variáveis derivadas | Python |
| 5.8 | Seleção por IV e correlação | Python |
| 5.9 | Checagem de qualidade | Python |
| 6.1 – 6.3 | Matriz de confusão, métricas, efeito do corte | Python |
| 6.4 – 6.6 | AUC, Gini, KS, gains table | Python |
| 6.7 | PSI | Python |
| 6.8 | Aderência | Python |
| 7.1 | Tabela de cortes | Python |
| 7.2 – 7.3 | Rating A–E e taxa por rating | Python |
| 7.4 | Motor de decisão | Python |
| 8.1 – 8.2 | Curva de vintage | SQL e Python |
| 8.3 | Matriz de rolagem | Python |
| 9.1 – 9.2 | k-means e árvore | Python |
| 10.1 – 10.2 | Champion-Challenger e tamanho de amostra | Python |
| 10.3 – 10.4 | Swap-in/out e cortes por segmento | Python |
| 10.5 | Matriz de limite | Python |
| 11.1 – 11.4 | Backtesting, CSI, recalibração, semáforo | Python |
| 12.1 – 12.3 | Registro do modelo, histórico, mapa de qualidade | Python |
| 13.1 – 13.2 | Base do painel e resumo mensal | SQL e Python |

---

## Glossário rápido

| Termo | Significado |
|---|---|
| **PD** | *Probability of Default*: probabilidade de inadimplência |
| **LGD** | *Loss Given Default*: % perdido quando há inadimplência (1 − recuperação) |
| **EAD** | *Exposure at Default*: saldo exposto no momento da inadimplência |
| **Perda esperada** | PD × LGD × EAD |
| **Safra** | Grupo de contratos ou de cooperados com a mesma data de referência |
| **OOT** | *Out-of-time*: validação numa safra posterior à de treino |
| **Lift** | Quantas vezes um grupo converte mais que a média da base |
| **Reject inference** | Técnicas para estimar o desempenho de quem foi recusado |
| **Override** | Decisão manual contrária à recomendada pelo modelo |
| **Overfitting** | Modelo que decorou o treino e não generaliza |
| **Rolagem (roll rate)** | % da carteira que passa para a faixa de atraso seguinte |
| **Singular** | Cooperativa de crédito de primeiro nível, que atende diretamente os cooperados |
