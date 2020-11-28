# from pypi
import numpy
import matplotlib.pyplot as pyplot
import seaborn

from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_hastie_10_2
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

X_all = numpy.random.randn(5000, 1)

y_all = (X_all[:, 0] > 0) * 2 - 1

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all,
						    test_size=0.5,
						    random_state=42)

model = DecisionTreeClassifier(max_depth=1)
model.fit(X_train, y_train)

print('Accuracy for a single decision stump: {}'.format(model.score(X_test, y_test)))


gbc_model = GradientBoostingClassifier(n_estimators=5000, learning_rate=0.01,
				 max_depth=3,
				 random_state=0)
gbc_model.fit(X_train, y_train)

print('Accuracy for Gradient Booing: {}'.format(gbc_model.score(X_test, y_test)))

y_pred = gbc_model.predict_proba(X_test)[:, 1]

print("Test logloss: {}".format(log_loss(y_test, y_pred)))


def compute_loss(y_true, predicted_probabilities):
    """applies sigmoid to predictions before calling log_loss

    Args:
     y_true: the actual classifications
     predicted_probabilities: probabilities that class was 1
    """
    return log_loss(y_true, sigmoid(predicted_probabilities))
    
def print_loss(cumulative_predictions, y_test):
    """prints the log-loss for the predictions

    Args:
     cumulative_predictions (numpy.Array): The cumulative predictions for the model
    """
    print(" - Logloss using all trees:           {}".format(
	compute_loss(y_test, cumulative_predictions[-1, :])))
    print(" - Logloss using all trees but last:  {}".format(
	compute_loss(y_test, cumulative_predictions[-2, :])))
    print(" - Logloss using all trees but first: {}".format(
	compute_loss(y_test, cumulative_predictions[-1, :] - cumulative_predictions[0, :])))
    return

gbc_cumulative_predictions = numpy.array(
    [x for x in gbc_model.staged_decision_function(X_test)])[:, :, 0]

print_loss(gbc_cumulative_predictions, y_test)

def plot_predictions(cumulative_predictions, identifier):
    """plots the cumulative predictions

    Args:
     identifier (str): something to identify the model
     cumulative_predictions: predictions from trees
    """
    figure = pyplot.figure(figsize=(10, 6))
    axe = figure.gca()
    axe.plot(cumulative_predictions[:, y_test == 1][:, 0])

    axe.set_title("({}) Score vs Trees".format(identifier))
    axe.set_xlabel('Number of Trees')
    label = axe.set_ylabel('Cumulative Decision Score')
    return

plot_predictions(gbc_cumulative_predictions, "eta=0.01")

big_eta = GradientBoostingClassifier(n_estimators=5000, learning_rate=8,
				     max_depth=3, random_state=0)
big_eta.fit(X_train, y_train)

y_pred = big_eta.predict_proba(X_test)[:, 1]
print("Test logloss: {}".format(log_loss(y_test, y_pred)))

print('Accuracy for Big Eta: {}'.format(big_eta.score(X_test, y_test)))

big_eta_cumulative_predictions = numpy.array(
    [x for x in big_eta.staged_decision_function(X_test)])[:, :, 0]

print_loss(big_eta_cumulative_predictions, y_test)

plot_predictions(big_eta_cumulative_predictions, "eta=8")


X_hastie, y_hastie = make_hastie_10_2(random_state=0)
X_train_hastie, X_test_hastie, y_train_hastie, y_test_hastie = train_test_split(
    X_hastie,
    y_hastie,
    test_size=0.5,
    random_state=42)

stump = DecisionTreeClassifier(max_depth=1)
stump.fit(X_train_hastie, y_train_hastie)

print('Accuracy for a single decision stump: {}'.format(
    stump.score(X_test_hastie, y_test_hastie)))

tree = DecisionTreeClassifier()
tree.fit(X_train_hastie, y_train_hastie)
print('Accuracy for the Decision Tree: {}'.format(
    tree.score(X_test_hastie, y_test_hastie)))

gbc2_model = GradientBoostingClassifier(n_estimators=5000, learning_rate=0.01,
					max_depth=3,
					random_state=0)
gbc2_model.fit(X_train_hastie, y_train_hastie)
y_pred = gbc2_model.predict_proba(X_test_hastie)[:, 1]

print('Accuracy for Gradient Boosting: {}'.format(
    gbc2_model.score(X_test_hastie, y_test_hastie)))

gbc2_cumulative_predictions = numpy.array(
    [x for x in gbc2_model.staged_decision_function(X_test_hastie)])[:, :, 0]
print_loss(gbc2_cumulative_predictions, y_test_hastie)

xg_model = XGBClassifier(n_estimators=5000, learning_rate=0.01)
#print(xg_model)
xg_model.fit(X_train_hastie, y_train_hastie)

print('Accuracy for XGBoost: {}'.format(xg_model.score(X_test_hastie, y_test_hastie)))

y_pred = xg_model.predict_proba(X_test_hastie)[:, 1]

# Attetion: This will not work

xg_cumulative_predictions = numpy.array(
    [x for x in xg_model.staged_decision_function(X_test_hastie)])[:, :, 0]
print_loss(xg_cumulative_loss)
