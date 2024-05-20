from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import mglearn as mg

X, y = mg.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
print("Test Set accuracy: {:.2f}".format(clf.score(X_test, y_test)))