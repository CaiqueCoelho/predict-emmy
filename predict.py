from __future__ import division
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.utils import class_weight
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler


def separateProducers(producer):
    producer = str(producer).replace(" and", ",")
    producer = producer.replace(" in association with", ",")
    return producer


def getMaxProducers(producer):
    totalProducers = len(str(producer).split(','))
    return totalProducers


def fit_and_predict(modelName, model, train_x, train_y):
    k = 10
    scores = cross_val_score(model, train_x, train_y, cv=k)
    hit_hate = np.mean(scores)

    msg = "Hit Rate from {0}: {1}".format(modelName, hit_hate)
    print(msg)

    return hit_hate


def gettingDistributionOfDatas():
    winners = int(list(Y_labels).count(1))
    print('Quantity winners: ' + str(winners))

    losers = int(list(Y_labels).count(0))
    print('Quantity losers: ' + str(losers))

    total = len(Y_labels)
    print('Total Quantity: ' + str(total))

    distribution_winners = int(winners/total * 100)
    distribution_losers = int(losers/total * 100)
    print('\nDistribution of winning data: ' + str(distribution_winners) + '%')
    print('Distribution of losing data: ' + str(distribution_losers) + '% \n')

    print('class_weight:')
    class_weight_count = {1: distribution_winners, 0: distribution_losers}
    print(class_weight_count)

    return class_weight_count


def convertYLabels(value):
    if value == 'True':
        return 1
    else:
        return 0

def getStaffName(name):
  name = (str(name).split(',')[0]).lower()
  return name

def getMaxNominees(name):

    dictNominees = {}
    maxNominees = []

    for row in df.iterrows():
        producers = set(str(row[1][name]).split(','))
        for i in producers:
            if i in dictNominees:
                dictNominees[i] = dictNominees[i] + 1
            else:
                dictNominees[i] = 1

    for row in df.iterrows():
        producers = set(str(row[1][name]).split(','))
        maxNominee = 0
        for i in producers:
            if(dictNominees[i] > maxNominee):
                maxNominee = dictNominees[i]
        maxNominees.append(maxNominee)

    nameMaxNominees = 'maxNominees_' + name
    df[nameMaxNominees] = maxNominees

# get dataset to train/test and to predict
df = pd.read_csv('datasets/the_emmy_awards.csv')
useStaff = True
category = 'Outstanding Drama Series'
try:
  del df['Unnamed: 7']
  del df['Unnamed: 8']
  del df['Unnamed: 9']
except Exception as e:
  print(e)

# check if exist any NaN values
# df.isnull().values.any()

# check if exist any NaN values
# df_to_predict.isnull().values.any()

# if exist any Nan values we need to handle with this
#imputer = Imputer(missing_values = np.nan, strategy = 'mean', axis = 0)
#df_to_predict_np_array =  np.array(df_to_predict)
#imputer = imputer.fit(df_to_predict_np_array[:, 4:])
# df_to_predict_np_array[:, 4:] = imputer.transform(X[:, :])

# trasnforming and creating new data
df['producer'] = df['producer'].apply(separateProducers)
df['totalProducers'] = df['producer'].apply(getMaxProducers)
df['totalStaffs'] = df['staff'].apply(getMaxProducers)

producersList = [[], [], [], [], [], [], [], [], []]

for row in df.iterrows():
    producers = row[1]['producer'].split(',')
    for i in range(0, 9):
        if(i+1 <= len(producers)):
            producersList[i].append(producers[i].lower())
        else:
            producersList[i].append(producers[0].lower())

df['producer1'] = producersList[0]
df['producer2'] = producersList[1]
df['producer3'] = producersList[2]
df['producer4'] = producersList[3]
df['producer5'] = producersList[4]
df['producer6'] = producersList[5]
df['producer7'] = producersList[6]
df['producer8'] = producersList[7]
df['producer9'] = producersList[8]

getMaxNominees('producer')
getMaxNominees('nominee')
getMaxNominees('company')
getMaxNominees('staff')

df = df.loc[df['category'] == category]

if(useStaff):
  df['staff'] = df['staff'].apply(getStaffName)

if useStaff:
  listDummies = ['nominee', 'company', 'staff', 'totalStaffs' 'producer1', 'producer2', 'producer3',
                'producer4', 'producer5', 'producer6', 'producer7', 'producer8', 'producer9']

  listColumns = ['nominee', 'company', 'staff', 'totalStaffs', 'totalProducers', 'producer1', 'producer2', 'producer3',
                'producer4', 'producer5', 'producer6', 'producer7', 'producer8', 'producer9',
                'maxNominees_producer', 'maxNominees_nominee', 'maxNominees_company', 'year', 'win']

  listColumnsX = ['nominee', 'company', 'staff', 'totalStaffs', 'totalProducers', 'producer1', 'producer2', 'producer3',
                  'producer4', 'producer5', 'producer6', 'producer7', 'producer8', 'producer9',
                  'maxNominees_producer', 'maxNominees_nominee', 'maxNominees_company']
else:
  listDummies = ['nominee', 'company', 'producer1', 'producer2', 'producer3',
                'producer4', 'producer5', 'producer6', 'producer7', 'producer8', 'producer9']

  listColumns = ['nominee', 'company', 'totalProducers', 'producer1', 'producer2', 'producer3',
                'producer4', 'producer5', 'producer6', 'producer7', 'producer8', 'producer9',
                'maxNominees_producer', 'maxNominees_nominee', 'maxNominees_company', 'year', 'win']

  listColumnsX = ['nominee', 'company', 'totalProducers', 'producer1', 'producer2', 'producer3',
                  'producer4', 'producer5', 'producer6', 'producer7', 'producer8', 'producer9',
                  'maxNominees_producer', 'maxNominees_nominee', 'maxNominees_company']

X = df[listColumns]

X_previous_years = X.loc[X['year'] < 2020]
X_actual_year = X.loc[X['year'] == 2020]

# getting the importants attributes from original dataset to our dataset to train and test
X_train_test = X_previous_years[listColumnsX]
Y_labels = X_previous_years['win']

# getting the importants attributes from original dataset to our dataset to predict
X_to_predict = X_actual_year[listColumnsX]
Y_to_predict = X_actual_year[listColumnsX]

# to improve our analysis, we will do a pre-processing to normalize the data
if useStaff:
  attributes_to_normalize = ['totalStaffs', 'totalProducers', 'maxNominees_producer', 'maxNominees_nominee', 'maxNominees_company']
else:
  attributes_to_normalize = ['totalProducers', 'maxNominees_producer', 'maxNominees_nominee', 'maxNominees_company']
X_train_robust = X_train_test.copy()

minMax = MinMaxScaler().fit(X_train_test[attributes_to_normalize])
X_train_robust[attributes_to_normalize] = minMax.transform(X_train_test[attributes_to_normalize])
X_to_predict[attributes_to_normalize] = minMax.transform(X_to_predict[attributes_to_normalize])

# categorical values treatment
X_train_dict = X_train_robust.to_dict(orient='records')  # turn each row as key-value
X_to_predict_dict = X_to_predict.to_dict(orient='records')

# DictVectorizer
dv_X = DictVectorizer(sparse=False).fit(X_train_dict)
# sparse = False makes the output is not a sparse matrix apply dv_X on X_dict
X_train = dv_X.transform(X_train_dict)
X_to_predict = dv_X.transform(X_to_predict_dict)

Y_labels = Y_labels.values
Y_to_predict = Y_to_predict.values

# print(Y_labels)

for i in range(0, len(Y_labels)):
    Y_labels[i] = convertYLabels(str(Y_labels[i]))

for i in range(0, len(Y_to_predict)):
    Y_to_predict[i] = convertYLabels(str(Y_to_predict[i]))

gettingDistributionOfDatas()

# Getting class_weight distribution
class_weight = class_weight.compute_class_weight(
    'balanced', np.unique(Y_labels), Y_labels)
class_weight_dict = {1: class_weight[0], 0: class_weight[1]}
print(class_weight_dict)

# for simplicity lets transform ours dataframes in arrays
X_train_test = X_train
X_to_predict = X_to_predict
Y_labels = Y_labels

print(X_train_test)

# Percentage train
percentage_train = 0.8
size_train = percentage_train * len(Y_labels)
train_data_X = X_train_test[:int(size_train)]
train_data_Y = Y_labels[:int(size_train)]

# Percentage test
test_data_X = X_train_test[int(size_train):]
test_data_Y = Y_labels[int(size_train):]

# Counting quantity of predictions as won
qt_candidates = len(X_to_predict)
candidate = [0] * qt_candidates
candidate0 = [0] * qt_candidates
results = {}

train_data_Y = train_data_Y.astype(int)
test_data_Y = test_data_Y.astype(int)

def predict_results(model, result):
    results[result] = model
    model.fit(train_data_X, train_data_Y)
    resultEmmy = model.predict(X_to_predict)
    print('Emmy 2020: ' + str(resultEmmy))
    print('')
    for i in range(qt_candidates):
        if(resultEmmy[i] == 1.0):
            candidate0[i] = candidate0[i] + 1
    if(result > 0.79):
        for i in range(qt_candidates):
            if(resultEmmy[i] == 1.0):
                candidate[i] = candidate[i] + 1
                
# Predict Adaboost
model = AdaBoostClassifier()
result = fit_and_predict("AdaBoostClassifier", model, train_data_X, train_data_Y)
predict_results(model, result)

# Predict SVC
model = svm.SVC()
result = fit_and_predict("SVC Grided", model, train_data_X, train_data_Y)
predict_results(model, result)


# Predict KNN
model = neighbors.KNeighborsClassifier()
result = fit_and_predict("KNN", model, train_data_X, train_data_Y)
predict_results(model, result)

# Predict Logistic Regression
model = LogisticRegression()
result = fit_and_predict("LogisticRegression", model,
                         train_data_X, train_data_Y)
predict_results(model, result)

# Predict Bagging
model = BaggingClassifier()
result = fit_and_predict("BaggingClassifier", model,
                         train_data_X, train_data_Y)
predict_results(model, result)

# Predict Gradient Boosting
model = GradientBoostingClassifier()
result = fit_and_predict("GradientBoostingClassifier",
                         model, train_data_X, train_data_Y)
predict_results(model, result)

# Predict Random Forest Grided
model = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=3)
result = fit_and_predict("RandomForestClassifier Grided",
                         model, train_data_X, train_data_Y)
predict_results(model, result)

# Predict Random Forest
model = RandomForestClassifier()
result = fit_and_predict("RandomForestClassifier Grided",
                         model, train_data_X, train_data_Y)
predict_results(model, result)

# Predict Random Forestclass_weight count
model = RandomForestClassifier(class_weight=class_weight_dict)
result = fit_and_predict(
    "RandomForestClassifier Grided and class_weight count", model, train_data_X, train_data_Y)
predict_results(model, result)

# Predict Random Forest class_weight balanced
model = RandomForestClassifier(class_weight='balanced')
result = fit_and_predict(
    "RandomForestClassifier Grided and class_weight balanced", model, train_data_X, train_data_Y)
predict_results(model, result)

# Predict Extra Tress
model = ExtraTreesClassifier()
result = fit_and_predict("ExtraTreesClassifier Grided",
                         model, train_data_X, train_data_Y)
predict_results(model, result)


# Predict DecisionTree Grided
model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
result = fit_and_predict("DecisionTree Grided", model,
                         train_data_X, train_data_Y)
predict_results(model, result)

# Predict DecisionTree
model = tree.DecisionTreeClassifier()
result = fit_and_predict("DecisionTree", model, train_data_X, train_data_Y)
predict_results(model, result)

# The effectiveness of the algorithm that kicks everything 0 or 1 or a single value
# Devolve a quantidade do maior elemento
base_hit = max(Counter(test_data_Y).values())
base_one = list(Y_labels).count(1)
base_zero = list(Y_labels).count(0)
hit_rate_base = 100.0 * base_hit / len(test_data_Y)
print("Hit rate based on validation data: %f" % hit_rate_base)

maximum = max(results)
winner = results[maximum]
print('\n\n')
print(winner)
print('\n\n')
winner.fit(train_data_X, train_data_Y)
result = winner.predict(test_data_X)

len_to_predict = len(train_data_Y)
hit_rate = metrics.accuracy_score(test_data_Y, result)

print("Better algorithm hit rate in the real world " + "was: "
      + str(hit_rate) + "% " + "from " + str(len_to_predict) + " elements\n\n")

if useStaff:
  name_and_films = X_actual_year[['nominee', 'staff']]
else:
  name_and_films = X_actual_year[['nominee']]
print(name_and_films)
print('\n')
winner.fit(train_data_X, train_data_Y)
winner_result = winner.predict(X_to_predict)
print('\nBest model predict:')
print(winner_result)
print(name_and_films.iloc[[winner_result.tolist().index(max(winner_result))]])

print('\nWithout accuracy validation:')
print(candidate0)
print(name_and_films.iloc[[candidate0.index(max(candidate0))]])
print("\n")
print('\nOnly if accuracy > 89%:')
print(candidate)
print(name_and_films.iloc[[candidate.index(max(candidate))]])
