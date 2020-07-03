import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import fbeta_score,accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, NuSVC, LinearSVC

### carregando dados
endereco_arquivo = 'sms.csv'
base = pd.read_csv(endereco_arquivo, encoding = "ISO-8859-1")

### infomacoes da base
#print()
#base.info()

#Faz o Stemming para reduzir palavras flexionadas
stemmer = SnowballStemmer("english")

#Limpa o texto removendo as pontuações e 'stop words' e deixa todas as letras minúsculas
def limparTexto(mensagem):
    
    mensagem = mensagem.translate(str.maketrans('', '', string.punctuation))
    palavras = [stemmer.stem(word) for word in mensagem.split() if word.lower() not in stopwords.words("english")]
    
    return " ".join(palavras)

#Aplica a função cleanText aos textos da nossa base
base["Full_Text"] = base["Full_Text"].apply(limparTexto)
print(base.head(10))

#Transformando o texto limpo em uma representação que um modelo de aprendizado de máquina possa entender
vec = TfidfVectorizer(encoding = "latin-1", strip_accents = "unicode", stop_words = "english")
features = vec.fit_transform(base["Full_Text"])
print(features.shape)

#Preparando a base para criar o método de predição
def encodeCategory(cat):
    if cat == "yes":
        return 1
    else:
        return 0
        
base["IsSpam"] = base["IsSpam"].apply(encodeCategory)


X_train, X_test, y_train, y_test = train_test_split(features, base["IsSpam"], stratify = base["IsSpam"], test_size = 0.2)

#Treinando os classificadores SVM e o Naive Bayes

modelo1 = MultinomialNB()
modelo2 = LinearSVC()

modelo1.fit(X_train, y_train)
modelo2.fit(X_train, y_train)

resultado1 = modelo1.predict(X_test)
resultado2 = modelo2.predict(X_test)
print('Naive Bayes')
print('Accuracy score: {}'.format(accuracy_score(y_test, resultado1)))
print('Precision score: {}'.format(precision_score(y_test, resultado1)))
print('Recall score: {}'.format(recall_score(y_test, resultado1)))
print('F1 score: {}'.format(f1_score(y_test, resultado1)))
print('F-Beta score: {}'.format(fbeta_score(y_test, resultado1, beta = 0.5)))
print('')
print('SMV')
print('Accuracy score: {}'.format(accuracy_score(y_test, resultado2)))
print('Precision score: {}'.format(precision_score(y_test, resultado2)))
print('Recall score: {}'.format(recall_score(y_test, resultado2)))
print('F1 score: {}'.format(f1_score(y_test, resultado2)))
print('F-Beta score: {}'.format(fbeta_score(y_test, resultado2, beta = 0.5)))


