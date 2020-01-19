import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from warnings import filterwarnings
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)

__author__ = 'Benyamin Jafari'
filterwarnings('ignore')


def init(apply_pca=False):
    df = pd.read_csv('dataset/blocks.csv')
    y = df.iloc[:, 0]  # select first column.
    y = LabelEncoder().fit_transform(y)  # Encoding labels to numbers.
    X = df.iloc[:, 1:]  # remove first column.

    if not apply_pca:
        return train_test_split(X, y, test_size=.1)
    else:
        pca_ = PCA(whiten=True)
        pca_.fit(X)
        X_pca = pca_.transform(X)
        return train_test_split(X_pca, y, test_size=.1)


def knn_predictor(k=None):
    """
    KNN method.
    :param k: k-neighbors value.
    :return: If `k` be None, it returns best `k` in training mode, else
    it returns the predict result for the test set by the chosen `k`.
    """
    if k is None:
        k_values = np.arange(1, 7)

        '''Cross validation to find the best number of neighbors(k).'''
        folds = 10  # number of folds.
        cv_scores = []
        for i, k in enumerate(k_values):
            knn_ = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)
            cv_scores.append(
                np.mean(cross_val_score(knn_, X_train, y_train, cv=folds)))

        best_k = cv_scores.index(max(cv_scores)) + 1
    else:
        best_k = k

    knn_ = KNeighborsClassifier(n_neighbors=best_k, metric='minkowski', p=2)
    knn_.fit(X_train, y_train)
    accuracy = knn_.score(X_test, y_test)

    return best_k, accuracy


def parametric_classifications():
    logreg = LogisticRegression(
        multi_class="multinomial",
        solver="newton-cg",  # I also tried by 'saga'
        # penalty='none' # I also use this penalty.
    )
    logreg.fit(X_train, y_train)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)

    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    logreg_acc = logreg.score(X_test, y_test)
    lda_acc = lda.score(X_test, y_test)
    qda_acc = qda.score(X_test, y_test)
    gnb_acc = gnb.score(X_test, y_test)

    return logreg_acc, lda_acc, qda_acc, gnb_acc


print('\n', 'Part a:'.center(70, '#'))
print('Initializing sets ...')
X_train, X_test, y_train, y_test = init()
print("All done.")


print('\n', 'Part b:'.center(70, '#'))
best_k, _ = knn_predictor()
print(f'The best K is: {best_k}')


print('\n', 'Part c:'.center(70, '#'))
average_table = pd.DataFrame(
    columns=['logistic regression', 'LDA', 'QDA', 'NBC', 'KNN', 'PCA-based'],
    index=range(1000))

print('It takes few minutes, please wait ...')
for i, _ in average_table.iterrows():
    X_train, X_test, y_train, y_test = init()
    # best_k, _ = knn_predictor()
    _, average_table.iloc[i, 4] = knn_predictor(k=best_k)
    logreg_acc, lda_acc, qda_acc, gnb_acc = parametric_classifications()
    average_table.iloc[i, 0] = logreg_acc
    average_table.iloc[i, 1] = lda_acc
    average_table.iloc[i, 2] = qda_acc
    average_table.iloc[i, 3] = gnb_acc

print(average_table.mean(axis=0))


print('\n', 'Part d:'.center(70, '#'))
X_train, X_test, y_train, y_test = init()

knn_ = KNeighborsClassifier(n_neighbors=best_k, metric='minkowski', p=2)
knn_.fit(X_train, y_train)
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
logreg = LogisticRegression(
    multi_class="multinomial",
    solver="newton-cg",  # I also tried by 'saga'
    # penalty='none' # I also use this penalty.
)
logreg.fit(X_train, y_train)

print(
    'LDA Confusion Matrix:\n',
    confusion_matrix(y_test, lda.predict(X_test)))
print(
    'QDA Confusion Matrix:\n',
    confusion_matrix(y_test, qda.predict(X_test)))
print(
    'NBC Confusion Matrix:\n',
    confusion_matrix(y_test, gnb.predict(X_test)))
print(
    'LogReg Confusion Matrix:\n',
    confusion_matrix(y_test, logreg.predict(X_test)))
print(
    'KNN Confusion Matrix:\n',
    confusion_matrix(y_test, knn_.predict(X_test)))

logreg_acc = logreg.score(X_test, y_test)
lda_acc = lda.score(X_test, y_test)
qda_acc = qda.score(X_test, y_test)
gnb_acc = gnb.score(X_test, y_test)
knn_acc = knn_.score(X_test, y_test)
print('LDA Accuracy: ', lda_acc)
print('QDA Accuracy: ', qda_acc)
print('NBC Accuracy: ', gnb_acc)
print('LogReg Accuracy: ', logreg_acc)
print('KNN Accuracy: ', knn_acc)
print('The best predictor in this case is Logistic Regression')


print('\n', 'Part e:'.center(70, '#'))
X_train_pca, X_test_pca, y_train, y_test = init(apply_pca=True)
logreg = LogisticRegression(
    multi_class="multinomial",
    solver="newton-cg")

cv_scores = []
for dim in range(1, X_train.shape[1] + 1):
    cv_scores.append(
        np.mean(cross_val_score(
            logreg, X_train_pca[:, :dim], y_train, cv=10)))

best_pca = (cv_scores.index(max(cv_scores)) + 1)
print(f"The best number of PC scores is {best_pca}")

# plt.bar(np.arange(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_,
#         color="blue",
#         edgecolor="Red")
# plt.show()
#
# logreg = LogisticRegression(
#     multi_class="multinomial",
#     solver="newton-cg")
# logreg.fit(X_train_pca[:, :2], y_train)
# logreg.score(X_test_pca[:, :2], y_test)


print('It takes few minutes, please wait ...')
logreg = LogisticRegression(
    multi_class="multinomial",
    solver="newton-cg")
for i, _ in average_table.iterrows():
    X_train_pca, X_test_pca, y_train, y_test = init(apply_pca=True)
    logreg.fit(X_train_pca[:, :best_pca], y_train)
    average_table.iloc[i, -1] = logreg.score(X_test_pca[:, :best_pca], y_test)

print(average_table.mean(axis=0))
