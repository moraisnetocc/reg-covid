import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from PIL import Image
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
import matplotlib.pyplot as plt



# variável preditora
from tensorflow import metrics

X = np.array([])
# variável alvo
y = np.array([])


class CustomLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.formula = None
        self.X = None
        self.y = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        self.X = X
        self.y = y
        soma_xy = sum(X * y)
        soma_x_ao_quadrado = sum(X * X)
        soma_x = sum(X)
        soma_y = sum(y)
        n = len(X)
        media_x = X.mean()
        media_y = y.mean()

        # build formula y = ax + b
        a = (soma_xy - n * media_x * media_y) / (soma_x_ao_quadrado - n * (media_x ** 2))
        b = media_y - (a * media_x)

        self.coef_ = np.array([b])
        self.intercept_ = np.array([a])

        self.formula = lambda _x: (a * _x) + b

    def predict(self, x):
        return list(map(self.formula, x))

    # fonte: https://edisciplinas.usp.br/pluginfile.php/1479289/mod_resource/content/0/regr_lin.pdf
    def sum_total_quadratic(self):
        median = self.y.mean()
        return sum((y - median) ** 2)

    def sum_error_quadratic(self):
        predicted = self.predict(x=self.X)
        return sum((self.y - predicted) ** 2)

    def regression_quadratic_sum(self):
        return self.sum_total_quadratic() - self.sum_error_quadratic()

    def score(self):
        return self.regression_quadratic_sum() / self.sum_total_quadratic()

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    def open_image(self, filepath, size):
        image = Image.open(filepath, 'r')
        image = image.resize((size, size), Image.ANTIALIAS)
        return self.rgb2gray(np.array(image.convert('RGB')))  # .tolist()

    def create_image_data_set(self, N, size):
        filepath = "/Users/moraisneto/PycharmProjects/covidAI/Images/"
        imagesX = []
        for i in range(1, N + 1):
            imagesX.append(self.open_image(filepath + "{}.png".format(i), size))
        print('Imagens carregadas')
        return imagesX

    def predict_x(self, images, size, N):
        image_predicted = [[0 for j in range(size)] for i in range(size)]

        for i in range(size):
            for j in range(size):
                line = []
                for k in range(N):
                    line.append(images[k][i][j])
                x_train = line[:N-2]
                y_train = line[1:N-1]
                self.fit(x_train, y_train)
                image_predicted[i][j] = self.predict([line[N-1]])[0]

        return np.asarray(image_predicted)



custom = CustomLinearRegression()
data = custom.create_image_data_set(14, 100)
result = custom.predict_x(data, 100, 13)
print(result)

teste = Image.fromarray(result)

# print('Mean Absolute Error:', metrics.mean_absolute_error(teste,  Image.fromarray(data[13])))

plt.imshow(teste)
plt.show()
print('Finalizado')
# custom.fit(X, y)
# print(custom.predict(X))

