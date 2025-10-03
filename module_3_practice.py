import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Загрузка данных
df = sns.load_dataset('titanic')

# Задача №1
# Преобразование столбца fare в массив NumPy и очистка от пропусков
fare_array = df['fare'].dropna().to_numpy()

# Расчет статистик
fare_mean = np.mean(fare_array)
fare_median = np.median(fare_array)
fare_std = np.std(fare_array)

print("Статистика стоимости билетов:")
print(f"Среднее: {fare_mean} ")
print(f"Медиана: {fare_median} ")
print(f"Стандартное отклонение: {fare_std}")


# Задача №2
# Создание таблицы количества мужчин и женщин по классам
gender_class_table = pd.crosstab(df['pclass'], df['sex'])
print("Распределение пассажиров по классам и полу:")
print(gender_class_table)



# Задача №3
# Заполнение пропусков средним возрастом по классу
df['age'] = df.groupby('pclass')['age'].transform(
    lambda x: x.fillna(x.mean())
)

# Создание возрастных групп
def age_group(age):
    if age < 18:
        return 'ребёнок'
    elif age <= 60:
        return 'взрослый'
    else:
        return 'пожилой'

df['age_group'] = df['age'].apply(age_group)

print("\nРаспределение по возрастным группам:")
print(df['age_group'].value_counts())



