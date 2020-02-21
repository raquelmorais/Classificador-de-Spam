import pandas as pd

#### QUESTAO 3 - Calcular Estatísticas


### carregando dados
endereco_arquivo = "C:/Users/Ana Raquel/Senior/desafio/sms_senior.csv"
base = pd.read_csv(endereco_arquivo, encoding = "ISO-8859-1")
### carregando dados

print("\nquestao 3: ")

# extraindo apenas os dois parâmetros
questao3 = pd.DataFrame({'Date': pd.to_datetime(base.Date), 'Word_Count': base.Word_Count})

# criando um novo parametro (mâs/ano)
questao3['Month/Year'] = questao3['Date'].apply(lambda x: "%d/%d" % (x.month, x.year))

# agrupa pelos dois parametros (o novo gerado e 'Word_Count')
# print(questao3)
agrupado_mes_q3 = questao3.groupby(['Month/Year'])

# algumas metricas do agrupamento
print(agrupado_mes_q3.describe())

print("\nmax: {}".format(agrupado_mes_q3.max().rename(columns={'Word_Count':'MAX'})))
print("\nmin: {}".format(agrupado_mes_q3.min().rename(columns={'Word_Count':'MIN'})))
print("\nmean: {}".format(agrupado_mes_q3.mean().rename(columns={'Word_Count':'MEAN'})))
print("\nmedian: {}".format(agrupado_mes_q3.median().rename(columns={'Word_Count':'MEDIAN'})))
print("\nstd: {}".format(agrupado_mes_q3.std().rename(columns={'Word_Count':'STD'})))
print("\nvar: {}".format(agrupado_mes_q3.var().rename(columns={'Word_Count':'VARIANCE'})))
print("\nsize: {}".format(agrupado_mes_q3.size()))

#### QUESTAO 3 - Calcular Estatísticas
