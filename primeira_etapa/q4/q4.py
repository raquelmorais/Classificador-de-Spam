import pandas as pd
import matplotlib.pyplot as plt

#### QUESTAO 4 - dia com mais mensagens comuns


### carregando dados
endereco_arquivo = "sms.csv"
base = pd.read_csv(endereco_arquivo, encoding = "ISO-8859-1")
base_comum = base.loc[base['IsSpam'] == 'comum']
### carregando dados

print("questao 4: ")

# extraindo apenas os dois parâmetros da base, com classificação 'comum'
questao4 = pd.DataFrame({'Date': pd.to_datetime(base_comum.Date), 'IsSpam': base_comum.IsSpam})

# criando dois novos parametros
questao4['Day/Month'] = questao4['Date'].apply(lambda x: "%d/%d" % (x.day, x.month))
questao4['Month'] = questao4['Date'].apply(lambda x: "%d" % (x.month))

# separando por mês
comum_mes_1 = questao4.loc[(questao4['Month'] == '1')]
comum_mes_2 = questao4.loc[(questao4['Month'] == '2')]
comum_mes_3 = questao4.loc[(questao4['Month'] == '3')]

# agrupando pelos dois parâmetros
agrupado_mes_1 = comum_mes_1.groupby(['Day/Month', 'IsSpam'])
agrupado_mes_2 = comum_mes_2.groupby(['Day/Month', 'IsSpam'])
agrupado_mes_3 = comum_mes_3.groupby(['Day/Month', 'IsSpam'])

# imprimindo o dia que teve a maior quantidade de spam no mês
# para ter lista ordenada por mês, basta tirar o método index
print("maior numero de msg comum de jan: {}".format(agrupado_mes_1.size().sort_values().index[-1]))
print("maior numero de msg comum de jan: {}".format(agrupado_mes_1.size().sort_values()))
#print("maior numero de msg comum de fev: {}".format(agrupado_mes_2.size().sort_values().index[-1]))
print("maior numero de msg comum de fev: {}".format(agrupado_mes_2.size().sort_values()))
#print("maior numero de msg comum de mar: {}".format(agrupado_mes_3.size().sort_values().index[-1]))
print("maior numero de msg comum de mar: {}".format(agrupado_mes_3.size().sort_values()))

#### QUESTAO 4 - dia com mais mensagens comuns
