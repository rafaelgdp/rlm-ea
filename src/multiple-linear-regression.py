# coding: utf-8

"""

Trabalho de Estatística Aplicada

    =========================
    Regressão Linear Múltipla
    =========================

    Grupo:
        Rafael G. de Pontes
        Henrique Arriel
        Jonathan L.

"""

import statsmodels.api as sm
import numpy as np
import pandas as pd
from pprint import *

predicted = 'TAXES'
predictor_columns = [
                    # 'TAXES',
                    'IBGE_POP',
                    # 'IBGE_RES_POP_BRAS',
                    # 'IBGE_RES_POP_ESTR',
                    # 'CAPITAL',
                    'IBGE_DU',
                    'PAY_TV',
                    'FIXED_PHONES',
                    # 'GDP',
                    # 'AREA',
                    # 'IBGE_1',
                    # 'IBGE_1-4',
                    # 'IBGE_5-9',
                    # 'IBGE_10-14',
                    # 'IBGE_15-59',
                    # 'IBGE_60+'
                    ]

# Organizando os dados para serem processados
brazil_cities_df = pd.read_csv('data/BRAZIL_CITIES.csv', error_bad_lines=False, sep=';', nrows=1200)
brazil_cities_df[predicted] = brazil_cities_df[predicted].str.replace(',', '.').astype(float)
for predictor in predictor_columns:
    if (not brazil_cities_df[predictor].dtype in ['int64']):
        brazil_cities_df[predictor] = brazil_cities_df[predictor].str.replace(',', '.').astype(float)
    print(predictor + " is " + str(brazil_cities_df[predictor].dtype))
target = pd.DataFrame(brazil_cities_df[predicted], columns=[predicted])
predictors = pd.DataFrame(brazil_cities_df[predictor_columns], columns=predictor_columns)

# X são as variáveis independentes com as quais se quer modelar a dependente
X = predictors[predictor_columns]
# y é a variável dependente que se quer prever
y = target[predicted]

# Construindo o modelo linear
model = sm.OLS(y, X).fit()
# Criando data frame com predições
predictions = model.predict(X)

pprint(model.summary())

# TODO: pensar em uma visualização (conjunto de gráficos) para comparar os dados com o modelo gerado
# TODO: repensar sobre as colunas escolhidas para o modelo