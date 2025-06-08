[Проект в Jupyter Notebook](https://github.com/DarsaGakaev/Pet_projects/blob/main/predicting_rating/cows.ipynb)

# Описание проекта
Для повышения уровня выпускников, министерство образования планирует предпринимать превентивные меры для улучшения оценок студентов. Для этого необходимо заранее выявлять потенциальных студентов, которые могут получить низкую итоговую оценку.В связи с этим необходимо:

- разработать модель, которая предскажет итоговую оценку ученика. Критерий успеха: метрика accuracy не ниже 95%.
  
# Навыки и инструменты
##### Обработка и анализ данных
import pandas as pd
import numpy as np
##### Визуализация
import matplotlib.pyplot as plt
import seaborn as sns
from phik.report import plot_correlation_matrix
##### Корреляционный анализ
from phik import phik_matrix
##### Предобработка данных
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
##### Модели машинного обучения
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
##### Обучение моделей и подбор гиперпараметров
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
##### Метрики качества
from sklearn.metrics import accuracy_score

# Общий вывод
Было проведено обучение указанной модели и достигнута требуемая метрика.
