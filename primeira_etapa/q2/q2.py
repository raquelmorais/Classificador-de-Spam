import pandas as pd
import matplotlib.pyplot as plt

#### QUESTAO 2 - classificacao por mês

### carregando dados
endereco_arquivo = "C:/Users/Ana Raquel/Senior/desafio/sms_senior.csv"
base = pd.read_csv(endereco_arquivo, encoding = "ISO-8859-1")

print("Questão 2: ")

# extraindo apenas os três parâmetros
questao2 = pd.DataFrame({'Date': pd.to_datetime(base.Date), 'Text': base.Full_Text, 'IsSpam': base.IsSpam})

# criando um novo parâmetro (mês/ano)
questao2['Month/Year'] = questao2['Date'].apply(lambda x: "%d/%d" % (x.month, x.year))
print(questao2)

# agrupa pelos dois parâmetros (o novo gerado e 'IsSpam')
agrupado_mes_q2 = questao2.groupby(['Month/Year', 'IsSpam'])

# agrega com o tamanho (.size) e já plota o grafico (.plot)
agrupado_mes_q2.size().plot(kind = 'pie', figsize = (6, 6), autopct = '%1.1f%%', label=' ', legend=True)
plt.title("Mensagens COMUNS vs SPAMS / por mês")
plt.legend(["JAN-Comum", "JAN-Spam", "FEV-Comum", "FEV-Spam", "MAR-Comum", "MAR-Spam"])
plt.show()


#### QUESTAO 2 - classificacao por mês
