from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, average_precision_score, recall_score, accuracy_score
import sys
import ast
import pandas as pd
import numpy as np
from sklearn import neighbors

# Helper Functions


def getLength(jsonListString):
    return len(ast.literal_eval(jsonListString))


def modifyDataSetForRegression(rawDataSet):
    rawDataSet.dropna(inplace=True)
    rawDataSet.drop(rawDataSet[rawDataSet['budget']
                               < 300000].index, inplace=True)

    rawDataSet = rawDataSet[['budget', 'runtime', 'cast', 'crew', 'genres']]

    rawDataSet.cast = rawDataSet.cast.apply(getLength)
    rawDataSet.genres = rawDataSet.genres.apply(getLength)
    rawDataSet.crew = rawDataSet.crew.apply(getLength)

    return rawDataSet


# Part 1
# Get the training data
training_data = pd.read_csv(sys.argv[1])
training_data.drop(
    training_data[training_data['revenue'] < 100000].index, inplace=True)
X_train = modifyDataSetForRegression(training_data)
y_train = training_data.revenue

# Get the test data
test_data = pd.read_csv(sys.argv[2])
test_data.drop(test_data[test_data['revenue'] < 100000].index, inplace=True)
X_test = modifyDataSetForRegression(test_data)
y_test = test_data.revenue

y_train = np.log1p(y_train)
y_test = np.log1p(y_test)

# Perform Linear Regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

pearsonCoeff = np.corrcoef(y_pred, y_test)[0, 1]

f = open("PART1.summary.csv", "w")
f.write("MSR,correlation\n")
f.write("%.2f,%.2f\n" % (mean_squared_error(y_test, y_pred), pearsonCoeff))
f.close()


def f(x):
    return np.int(x)


f2 = np.vectorize(f)

df = pd.DataFrame({'movie_id': test_data.movie_id,
                   'predicted_revenue': f2(np.exp(y_pred))})
df.to_csv(r'./PART1.output.csv', index=False)


# Part 2
# Get the training data
training_data = pd.read_csv(sys.argv[1])
X_train = training_data[['budget', 'runtime']]
y_train = training_data.rating


# Get the test data
test_data = pd.read_csv(sys.argv[2])
X_test = test_data[['budget', 'runtime']]
y_test = test_data.rating

# Perform Nearest Neighbour Classification
clf = neighbors.KNeighborsClassifier(17, weights='distance')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

avg_precision = average_precision_score(
    np.array(y_pred), np.array(y_test), pos_label=2)
recall = recall_score(np.array(y_pred), np.array(y_test), pos_label=2)
accuracy = accuracy_score(np.array(y_pred), np.array(y_test))

f = open("PART2.summary.csv", "w")
f.write("average_precision,average_recall,accuracy\n")
f.write("%.2f,%.2f,%.2f\n" % (avg_precision, recall, accuracy))
f.close()

df = pd.DataFrame({'movie_id': test_data.movie_id, 'predicted_rating': y_pred})
df.to_csv(r'./PART2.output.csv', index=False)
