import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.linear_model import LogisticRegression
from scipy.fftpack import dct, idct
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import metrics


class RegressionTree:
    N = 10
    imagesX = []
    size = 200

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    # implement 2D DCT
    def dct2(self, a):
        return dct(dct(a.T, norm='ortho').T, norm='ortho')

    # implement 2D IDCT
    def idct2(self, a):
        return idct(idct(a.T, norm='ortho').T, norm='ortho')

    def __open_image(self, filepath):
        image = Image.open(filepath, 'r')
        image = image.resize((self.size, self.size), Image.ANTIALIAS)
        return self.rgb2gray(np.array(image.convert('RGB')))

    def create_image_data_set(self):
        filepath = "/Users/moraisneto/PycharmProjects/covidAI/ImagesV2/"
        for i in range(1, self.N+1):
            self.imagesX.append(self.__open_image(filepath + "{}.png".format(i)))
        print('Imagens carregadas')

    def predict_with_logistic(self):
        regressor = DecisionTreeRegressor()
        for i in range(0, self.N-2):
            regressor.fit(self.imagesX[i], self.imagesX[i+1])
            print("Fit {} completed".format(i))
            # print(regressor.feature_importances_)
        return regressor.predict(self.imagesX[8])

    def predict_with_random_subspace(self):
        regressor = BaggingRegressor(random_state=42, n_estimators=200, bootstrap_features=True)
        for i in range(0, self.N-2):
            regressor.fit(self.imagesX[i], self.imagesX[i+1])
            print("Fit {} completed".format(i))
            # print(regressor.feature_importances_)
        return regressor.predict(self.imagesX[8])

    def predict_with_svr(self):
        regressor = svm.SVC(kernel="linear")
        for i in range(0, self.N-2):
            regressor.fit(self.imagesX[i], self.imagesX[i+1])
            print("Fit {} completed".format(i))
            # print(regressor.feature_importances_)
        return regressor.predict(self.imagesX[8])

    def predict_with_random_forest(self):
        regressor = RandomForestRegressor(random_state=42, oob_score=True)
        for i in range(0, self.N-2):
            regressor.fit(self.imagesX[i], self.imagesX[i+1])
            print("Fit {} completed".format(i))
            # print(regressor.feature_importances_)
        return regressor.predict(self.imagesX[8])

    def predict_with_single_tree(self):
        regressor = DecisionTreeRegressor(random_state=42)
        for i in range(0, self.N-1):
            regressor.fit(self.imagesX[i], self.imagesX[i+1])
        return regressor.predict(self.imagesX[self.N - 1])

    def predict_with_single_tree_v2(self):
        regressor = DecisionTreeRegressor(random_state=42)
        return cross_val_score(regressor, self.imagesX[:int(self.N-1)], self.imagesX[1:], cv=2)

    def __predict_two(self, regressor):
        part = self.size / 2
        for i in range(0, self.N-2):
            regressor.fit(self.imagesX[i][:int(part)], self.imagesX[i+1][:int(part)])
        parte1 = regressor.predict(self.imagesX[self.imagesX.__len__()-1][:int(part)])

        regressor = BaggingRegressor(random_state=42)
        for i in range(0, self.N - 2):
            regressor.fit(self.imagesX[i][int(part+1):], self.imagesX[i+1][int(part+1):])
        parte2 = regressor.predict(self.imagesX[self.imagesX.__len__()-1][int(part+1):])
        image = np.concatenate((parte1, parte2))
        return image

    def predict_with_two_forests(self):
        regressor = BaggingRegressor(random_state=42)
        return self.__predict_two(regressor)

    def predict_with_two_subspaces(self):
        regressor = BaggingRegressor(random_state=42)
        return self.__predict_two(regressor)

    def predict_with_n_subspaces(self, amount):
        amount_array = [int(x) for x in range( 0, self.size, int(self.size/amount))]
        image = []
        for i in range(1, amount_array.__len__()):
            # regressor = RandomForestRegressor(random_state=42, n_estimators=100, n_jobs=-1, oob_score=True)
            regressor = BaggingRegressor(random_state=42, n_estimators=100, n_jobs=-1)
            X = []
            y = []
            if i == amount_array.__len__() - 1:
                for j in range(0, self.N - 2):
                    X = self.imagesX[j][int(amount_array[i]):]
                    y = self.imagesX[j + 1][int(amount_array[i]):]
                    regressor.fit(X,y)
                image = np.concatenate(
                    (
                        image,
                        regressor.predict(y)
                    )
                )
                break
            if i == 1:
                for j in range(0, self.N - 2):
                    X = self.imagesX[j][:int(amount_array[i])]
                    y = self.imagesX[j+1][:int(amount_array[i])]
                    regressor.fit(X,y)
                image = regressor.predict(y)
            else:
                for j in range(0, self.N - 2):
                    X = self.imagesX[j][amount_array[i - 1]:amount_array[i]]
                    y = self.imagesX[j + 1][amount_array[i - 1]:amount_array[i]]
                    regressor.fit(X,y)
                image = np.concatenate(
                    (
                        image,
                        regressor.predict(y)
                    )
                )

        return image

    def predict_with_m_n_trees(self, amount):
        amount_array = [int(x) for x in range( 0, self.size, int(self.size/amount))]
        image = []
        for i in range(1, amount_array.__len__()):
            regressor = DecisionTreeRegressor(random_state=42)
            X = []
            y = []
            if i == 1:
                for j in range(0, self.N - 1):
                    X = self.imagesX[j][:int(amount_array[i])].copy()
                    y = self.imagesX[j+1][:int(amount_array[i])].copy()
                    X = [row[:amount_array[i]] for row in X]
                    y = [row[:amount_array[i]] for row in y]
                    regressor.fit(
                        X,y
                    )
                image = regressor.predict(y)
            if i == amount_array.__len__() - 1:
                for j in range(0, self.N - 1):
                    X = self.imagesX[j][int(amount_array[i]):]
                    y = self.imagesX[j+1][int(amount_array[i]):]
                    X = [row[amount_array[i]:] for row in X]
                    y = [row[amount_array[i]:] for row in y]
                    regressor.fit(
                        X,y
                    )
                image = np.concatenate(
                    (
                        image,
                        regressor.predict(y)
                    )
                )
            else:
                for j in range(0, self.N - 1):
                    X = self.imagesX[j][amount_array[i-1]:int(amount_array[i])].copy()
                    y = self.imagesX[j+1][amount_array[i-1]:int(amount_array[i])].copy()
                    X = [row[amount_array[i-1]:amount_array[i]] for row in X]
                    y = [row[amount_array[i-1]:amount_array[i]] for row in y]
                    regressor.fit(
                        X,y
                    )
                image = np.concatenate(
                    (
                        image,
                        regressor.predict(y, check_input=False)
                    )
                )

        return image

    def predict_pixel_to_pixel(self):
        image = []
        for i in range(0, self.size):
            row = []
            for j in range(0, self.size):
                if j == 0:
                    row = self.predict_line(i, j)
                else:
                    row = np.concatenate(
                        (
                            row,
                            self.predict_line(i, j)
                        )
                    )
            if i == 0:
                image = [row]
            else:
                image.append(row)
        return np.asarray(image)

    def predict_line(self, row, col):
        regressor = DecisionTreeRegressor(random_state=42)
        sum = 0
        for image in range(self.N-2):
            sum += self.imagesX[image][row][col]
            regressor.fit(
                [[self.imagesX[image][row][col]]],
                [[self.imagesX[image + 1][row][col]]]
            )
        sum += self.imagesX[self.N-1][row][col]
        return regressor.predict([[sum/self.N]])

    def get_last_image(self):
        return np.asarray(self.imagesX[self.N-1])

    def get_image(self, index):
        return np.asarray(self.imagesX[index])

    def predict_n_n(self):
        image_predicted = [[0 for j in range(self.N)] for i in range(self.N)]

        for i in range(self.size):
            for j in range(self.size):
                regressor = DecisionTreeRegressor(random_state=42)
                for k in range(0, self.N-1):
                    regressor.fit([self.imagesX[k][i][j]], [self.imagesX[k+1][i][j]])
                image_predicted[i][j] = regressor.predict(self.imagesX[self.N-1][i][j])

        return np.asarray(image_predicted)


r = RegressionTree()
r.create_image_data_set()

original = Image.fromarray(r.get_image(9))
plt.imshow(original)
plt.show()

image_predicted = r.predict_n_n()
# image_predicted = r.predict_with_random_subspace()
# image_predicted = r.predict_with_logistic()
# image_predicted = r.predict_with_two_subspaces()
# image_predicted = r.predict_with_n_subspaces(200)
# image_predicted = r.predict_with_svr()
teste = Image.fromarray(image_predicted)

# print('Mean Absolute Error:', metrics.mean_absolute_error(image_predicted, r.get_image(14)))
plt.imshow(teste)
plt.show()
print('Finalizado')

# image2 = r.predict_with_n_subspaces(200)
# plt.imshow(Image.fromarray(image2))
# print('Finalizado')
