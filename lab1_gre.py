import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from IPython.core.display import display

warnings.filterwarnings("ignore")

training_data = pd.read_csv('data/lab1_gre/training.csv')
test_data = pd.read_csv('data/lab1_gre/test_features.csv')
test_data['Time at Center'] = pd.read_csv('data/lab1_gre/y_test.csv', header=None).values
print('The shape of the training dataset is:', training_data.shape)
print('The shape of the test dataset is:', test_data.shape)

# Получим количество строк, столбцов и некоторую простую статистику набора данных.
# Реализация здесь
print(training_data.info())
print(training_data.head())

# Реализация здесь
print(test_data.info())
print(test_data.head())
# Выбор признаков для построения модели
numerical_features = ['Age upon Intake Days']
categorical_features = [
    'Outcome Type', 'Sex upon Outcome', 'Intake Type',
    'Intake Condition', 'Pet Type',
    'Sex upon Intake'
]
model_target = 'Time at Center'

# Обработка данных
print(training_data[numerical_features].isna().sum())
print(training_data[categorical_features].isna().sum())
print(test_data[numerical_features].isna().sum())
print(test_data[categorical_features].isna().sum())


class ProviderData:
    def __init__(self, X, y):
        self.X = X
        self.y = y


class ScalerProvider:

    def __init__(self, scaler, classifier, training_data, test_data):
        self.scaler = scaler
        self.classifier = classifier
        self.training_data = training_data
        self.test_data = test_data

    def fill_data(self, numerical_features, categorical_features):
        number_train = self.training_data[numerical_features].copy()
        categorical_train = self.training_data[categorical_features].copy()

        number_test = self.test_data[numerical_features].copy()
        categorical_test = self.test_data[categorical_features].copy()

        # Бинарое кодирование категориальных признаков
        X_train_categorical = pd.get_dummies(categorical_train)
        X_test_categorical = pd.get_dummies(categorical_test)

        X_train_numerical = self.scaler.fit_transform(number_train.values)
        X_test_numerical = self.scaler.transform(number_test)

        X_train = np.hstack((X_train_categorical.values, X_train_numerical))
        y_train = self.training_data[model_target].values

        X_test = np.hstack((X_test_categorical.values, X_test_numerical))
        y_test = self.test_data[model_target].values

        return ProviderData(X=X_train, y=y_train), ProviderData(X=X_test, y=y_test)


clear_lambda = lambda df: df.copy().drop(columns=['Name'], axis=1).dropna()
clear_training_data = training_data.apply(clear_lambda)
clear_test_data = test_data.apply(clear_lambda)

provider = ScalerProvider(
    scaler=StandardScaler(),
    classifier=LogisticRegression(random_state=0),
    training_data=clear_training_data,
    test_data=clear_test_data
)

train_provider_data, test_provider_data = provider.fill_data(
    numerical_features=numerical_features,
    categorical_features=categorical_features
)

#Обучение модели
provider.classifier.fit(train_provider_data.X, train_provider_data.y)
train_predicted = provider.classifier.predict(train_provider_data.X)
print(f'Train accuracy: {accuracy_score(train_provider_data.y, train_predicted)}')

# Прогнозирование на тестовом наборе данных
test_predicted = provider.classifier.predict(test_provider_data.X)
print(test_predicted)
print(f'Test accuracy: {accuracy_score(test_provider_data.y, test_predicted)}')

# Запись результатов прогнозирования в CSV-файл
pd.DataFrame(test_predicted).to_csv('data//result.csv', header=False, index=False)
