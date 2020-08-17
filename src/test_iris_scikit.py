from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

accuracy_score = clf.score(X_test, y_test)
print('Accuracy Score: ', accuracy_score)

dot_data = tree.export_graphviz(clf, out_file='data/output_test.dot')
