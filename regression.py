import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


weather_df = pd.read_csv('/kaggle/input/weather-prediction/seattle-weather.csv')

import missingno as msno
import seaborn as sns 
import matplotlib.pyplot as plt


weather_df.info()

weather_df.head()

weather_df.hist(figsize=(10,8))

le = LabelEncoder()

encoded = le.fit_transform(weather_df['weather'])

weather_df['encoded_weather'] = encoded

weather_df.head()

sns.heatmap(weather_df.corr())

g = sns.PairGrid(weather_df, y_vars='encoded_weather', 
                x_vars=['precipitation', 'temp_max', 'temp_min', 'wind'], height=4)
g.map(sns.regplot)