from urllib import request
import tarfile
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt


# Download file
print("Downloading dataset ...")
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00401/TCGA-PANCAN-HiSeq-801x20531.tar.gz"
response = request.urlretrieve(URL, "data/TCGA-PANCAN-HiSeq-801x20531.tar.gz")

# open file
print("Extracting files ...")
file = tarfile.open('data/TCGA-PANCAN-HiSeq-801x20531.tar.gz')
# extract files
file.extractall('data/')
file.close()

#########################################
#            Data loading               #
#########################################
data = pd.read_csv('data/TCGA-PANCAN-HiSeq-801x20531/data.csv', index_col=0)
labels = pd.read_csv('data/TCGA-PANCAN-HiSeq-801x20531/labels.csv', index_col=0)

#########################################
#    Data preprocessing and cleaning    #
#########################################

# Remove columns/features with all zeros
data = data.loc[:, (data != 0).any(axis=0)]
# Remove rows/samples with all zeros
data = data[~(data == 0).all(axis=1)]
# Keep only the labels of the samples remaining in the dataset
labels = labels.loc[data.index]

print(f"number of features left : {data.shape[1]}")
print(f"number of samples left : {data.shape[0]}")

#########################################
#            Data splitting             #
#########################################
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

#########################################
#            Model Training             #
#########################################
# Logistic regression
logreg = linear_model.LogisticRegression(max_iter=100000000, verbose=1)
logreg_parameters = {'C': [1e-40, 1e-20, 0.001, 1, 1e3, 1e20, 1e40]}
logreg_grid_search = GridSearchCV(estimator=logreg,
                                  param_grid=logreg_parameters,
                                  scoring='accuracy',
                                  cv=5,
                                  verbose=1)

logreg_grid_search = logreg_grid_search.fit(X_train, Y_train)

# SVM
SVM = SVC()
SVM_parameters = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}
SVM_grid_search = GridSearchCV(SVM, SVM_parameters, refit=True, scoring='accuracy', cv=5, verbose=1)
SVM_grid_search = SVM_grid_search.fit(X_train, Y_train)

# Random Forest
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 5],
    'min_samples_split': [8, 12],
    'n_estimators': [100, 500]
}

RF = RandomForestClassifier()
RF_grid_search = GridSearchCV(estimator=RF, param_grid=param_grid,
                              cv=5, scoring='accuracy', verbose=1)
RF_grid_search = RF_grid_search.fit(X_train, Y_train)

#########################################
#            Model Evaluation           #
#########################################

# Metrics
models = {"Multiclass logistic regression": logreg_grid_search, "SVM": SVM_grid_search, "Random Forest": RF_grid_search}
for model_name, model in models.items():
    print(model_name)
    test_prediction = model.predict(X_test)
    train_prediction = model.predict(X_train)
    report = classification_report(Y_test, test_prediction)
    print(report)
    print("Best parameters : ", model.best_params_)
    print("Training accuracy : ", accuracy_score(Y_train, train_prediction))
    print()
    with open(f"output/{'_'.join(model_name.split(' '))}.txt", 'w') as file:
        file.writelines(report + "\n")
        file.write("Best parameters : " + str(model.best_params_) + "\n\n")
        file.write("Training accuracy : " + str(accuracy_score(Y_train, train_prediction)))

# Learning curve

def learning_curve(model, model_parameters, title):
    training_accuracy = []
    testing_accuracy = []

    index = [50, 100, 500, 800]
    for i in index:
        grid_search = GridSearchCV(estimator=model,
                                   param_grid=model_parameters,
                                   scoring='accuracy',
                                   cv=5,
                                   verbose=1)
        grid_search.fit(X_train.iloc[1:i, :], Y_train[1:i])
        training_accuracy.append(accuracy_score(Y_train[1:i], grid_search.predict(X_train.iloc[1:i, :])))
        testing_accuracy.append(accuracy_score(Y_test, grid_search.predict(X_test)))

    plt.plot(index, training_accuracy, label='Training', color='b')  # plotting t, a separately
    plt.plot(index, testing_accuracy, label='Testing', color='g')  # plotting t, b separately
    plt.title(title)
    plt.xlabel("Number of samples")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"output/{'_'.join(title.split(' '))}.png")
    plt.show()
    plt.clf()


# Logistic regression
logreg = linear_model.LogisticRegression(max_iter=100000000)
logreg_parameters = {'C': [0.001]}
# SVM
SVM = SVC()
SVM_parameters = {'C': [10], 'gamma': [0.0001], 'kernel': ['rbf']}
# RF
RF = RandomForestClassifier()
RF_parameters = {'bootstrap': [True], 'max_depth': [110], 'max_features': [3],
                 'min_samples_leaf': [3], 'min_samples_split': [12], 'n_estimators': [500]}

learning_curve(logreg, logreg_parameters, "Multiclass logistic regression learning curve")
learning_curve(SVM, SVM_parameters, "SVM learning curve")
learning_curve(RF, RF_parameters, "Random Forest learning curve")
