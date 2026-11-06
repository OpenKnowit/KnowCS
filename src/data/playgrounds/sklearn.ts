// scikit-learn 实验台：课程里的模型、预处理和评估
import { MATPLOTLIB } from '../../lib/pyPlot'
import { SKLEARN } from '../../lib/pySklearn'
import type { PlayConfig, PlayEntry } from './types'

const ENTRIES: PlayEntry[] = [
  {
    id: 'knn', cat: 'models', label: 'KNeighborsClassifier',
    code: `import numpy as np
from sklearn.neighbors import KNeighborsClassifier

X = np.array([[0, 0], [1, 1], [8, 8], [9, 9]])
y = np.array([0, 0, 1, 1])
knn = KNeighborsClassifier(n_neighbors=3).fit(X, y)
print(knn.predict([[2, 2], [7, 7]]))
print(knn.predict_proba([[2, 2]]))
print(knn.kneighbors([[2, 2]], n_neighbors=2)[1])
print(knn.score(X, y))`,
  },
  {
    id: 'naive_bayes', cat: 'models', label: 'GaussianNB',
    code: `import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[1.], [2.], [3.], [7.], [8.], [9.]])
y = np.array([0, 0, 0, 1, 1, 1])
nb = GaussianNB().fit(X, y)
print(nb.theta_.ravel(), np.round(nb.var_.ravel(), 4), nb.class_prior_)
print(nb.predict([[4.], [6.]]))
print(np.round(nb.predict_proba([[5.]]), 4))`,
  },
  {
    id: 'kmeans', cat: 'models', label: 'KMeans',
    code: `import numpy as np
from sklearn.cluster import KMeans

X = np.array([[0., 0.], [0., 1.], [9., 9.], [9., 8.]])
km = KMeans(n_clusters=2, n_init=1, init=np.array([[0., 0.], [9., 9.]])).fit(X)
print(km.labels_, km.cluster_centers_, km.inertia_, km.n_iter_)
print(km.predict([[1., 1.]]))`,
  },
  {
    id: 'linear', cat: 'models', label: 'LinearRegression',
    code: `import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

X = np.array([[1.], [2.], [3.], [4.]])
y = np.array([3., 5., 7., 9.])
lr = LinearRegression().fit(X, y)
print(lr.coef_, lr.intercept_, lr.predict([[5.]]), lr.score(X, y))
print(Ridge(alpha=1.0).fit(X, y).coef_)`,
  },
  {
    id: 'tree', cat: 'models', label: 'DecisionTreeClassifier',
    code: `import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
t = DecisionTreeClassifier(random_state=0).fit(X, y)
print(t.score(X, y), t.get_depth())
print(export_text(t, feature_names=['a', 'b']))`,
  },
  {
    id: 'pipeline', cat: 'prep', label: 'Pipeline',
    code: `import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

X = np.array([[1., 100.], [2., 300.], [3., 200.]])
s = StandardScaler().fit(X)
print(s.mean_, s.scale_, s.transform(X).round(3))
print(MinMaxScaler().fit_transform(X).round(3))
pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(1)).fit(X, [0, 1, 0])
print(pipe.predict([[2., 250.]]), list(pipe.named_steps))`,
  },
  {
    id: 'encode', cat: 'prep', label: 'OneHotEncoder',
    code: `import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures

print(LabelEncoder().fit_transform(['cat', 'dog', 'cat']))
one = OneHotEncoder(sparse_output=False).fit([['red'], ['blue']])
print(one.categories_[0], one.transform([['red']]))
print(PolynomialFeatures(2).fit_transform([[2., 3.]]))`,
  },
  {
    id: 'metrics', cat: 'eval', label: 'classification metrics',
    code: `import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

y = [1, 0, 1, 1, 0, 1]
p = [1, 0, 0, 1, 1, 1]
print(confusion_matrix(y, p))
print(accuracy_score(y, p), precision_score(y, p), recall_score(y, p), f1_score(y, p))
print(classification_report(y, p))`,
  },
  {
    id: 'cv', cat: 'eval', label: 'cross_val_score',
    code: `import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier

X = np.arange(20.).reshape(10, 2)
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
print([t.tolist() for _, t in KFold(5).split(X)][:2])
print(cross_val_score(KNeighborsClassifier(1), X, y, cv=KFold(5, shuffle=True, random_state=0)).shape)
print([a.shape for a in train_test_split(X, y, test_size=0.3, random_state=0)])
g = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': [1, 3]}, cv=KFold(2, shuffle=True, random_state=0)).fit(X, y)
print(sorted(g.best_params_), g.cv_results_['mean_test_score'].shape)`,
  },
  {
    id: 'pca', cat: 'eval', label: 'PCA',
    code: `import numpy as np
from sklearn.decomposition import PCA

X = np.array([[1., 2.], [2., 4.1], [3., 5.9], [4., 8.]])
p = PCA(n_components=1).fit(X)
print(np.round(p.explained_variance_ratio_, 4), np.round(np.abs(p.components_), 3), np.round(p.mean_, 3))
print(np.abs(p.transform(X)).round(3).ravel())`,
  },
  // ---- the everyday APIs; each one was run on the real library
  {
    id: 'iris', cat: 'data', label: 'load_iris',
    code: `import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
print(iris.data.shape, iris.target.shape)
print(iris.feature_names)
print(iris.target_names)
print(iris.data[:2], iris.target[:2])
print(np.bincount(iris.target))`,
  },
  {
    id: 'make_data', cat: 'data', label: 'make_blobs / make_classification',
    code: `import numpy as np
from sklearn.datasets import make_blobs, make_classification
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=90, centers=3, n_features=2, random_state=0)
print(X.shape, y.shape, np.bincount(y))
A, b = make_classification(n_samples=50, n_features=4, n_informative=2, random_state=0)
print(A.shape, b.shape)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()`,
  },
  {
    id: 'workflow', cat: 'data', label: 'split → scale → fit → score',
    code: `import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
print(X_train.shape, X_test.shape, np.bincount(y_test))
scaler = StandardScaler().fit(X_train)
model = LogisticRegression(max_iter=500).fit(scaler.transform(X_train), y_train)
pred = model.predict(scaler.transform(X_test))
print(pred.shape, accuracy_score(y_test, pred) > 0.8)`,
  },
  {
    id: 'logistic', cat: 'models', label: 'LogisticRegression',
    code: `import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([[1.], [2.], [3.], [4.], [5.], [6.]])
y = np.array([0, 0, 0, 1, 1, 1])
m = LogisticRegression().fit(X, y)
print(m.classes_, m.coef_.shape, m.intercept_.shape)
print(m.predict([[1.], [6.]]))
print(np.round(m.predict_proba([[3.5]]), 3))
print(m.score(X, y))`,
  },
  {
    id: 'ridge_perceptron', cat: 'models', label: 'Ridge / Perceptron',
    code: `import numpy as np
from sklearn.linear_model import Perceptron, Ridge

X = np.array([[1.], [2.], [3.], [4.]])
y = np.array([2., 4., 6., 8.])
for alpha in [0.01, 1., 10.]:
    r = Ridge(alpha=alpha).fit(X, y)
    print(alpha, np.round(r.coef_, 3), np.round(r.intercept_, 3))
p = Perceptron(random_state=0).fit([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 0, 0, 1])
print(p.predict([[0, 0], [1, 1]]), p.coef_.shape)`,
  },
  {
    id: 'knn_reg', cat: 'models', label: 'KNeighborsRegressor',
    code: `import numpy as np
from sklearn.neighbors import KNeighborsRegressor

X = np.array([[1.], [2.], [3.], [10.], [11.]])
y = np.array([1., 2., 3., 10., 11.])
for k, w in [(1, 'uniform'), (3, 'uniform'), (3, 'distance')]:
    m = KNeighborsRegressor(n_neighbors=k, weights=w).fit(X, y)
    print(k, w, np.round(m.predict([[2.5], [9.]]), 3))`,
  },
  {
    id: 'nb_family', cat: 'models', label: 'MultinomialNB / BernoulliNB',
    code: `import numpy as np
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, MultinomialNB

# word counts for four short messages: ham, ham, spam, spam
X = np.array([[3, 0, 1], [2, 0, 0], [0, 4, 1], [0, 3, 2]])
y = np.array([0, 0, 1, 1])
m = MultinomialNB().fit(X, y)
print(m.predict([[1, 0, 0], [0, 2, 0]]), np.round(m.predict_proba([[1, 1, 0]]), 3))
print(np.round(np.exp(m.feature_log_prob_), 3))
print(BernoulliNB().fit(X > 0, y).predict([[True, False, False]]))
print(CategoricalNB().fit([[0, 1], [0, 0], [1, 1], [1, 0]], y).predict([[1, 1]]))`,
  },
  {
    id: 'tree_text', cat: 'models', label: 'export_text',
    code: `import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text

iris = load_iris()
t = DecisionTreeClassifier(max_depth=2, random_state=0).fit(iris.data, iris.target)
print(export_text(t, feature_names=list(iris.feature_names)))
print(t.get_depth(), t.get_n_leaves(), np.round(t.feature_importances_, 3))`,
  },
  {
    id: 'roc', cat: 'eval', label: 'roc_curve / roc_auc_score',
    code: `import numpy as np
from sklearn.metrics import auc, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

y = np.array([0, 0, 1, 1, 0, 1, 1, 0])
scores = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.6, 0.5])
fpr, tpr, thresholds = roc_curve(y, scores)
print(np.round(fpr, 2), np.round(tpr, 2))
print(auc(fpr, tpr), roc_auc_score(y, scores))
plt.plot(fpr, tpr, marker='o')
plt.plot([0, 1], [0, 1], '--')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.show()`,
  },
  {
    id: 'regression_metrics', cat: 'eval', label: 'mean_squared_error / r2_score',
    code: `import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

y = [3., 5., 2., 7.]
p = [2.5, 5., 3., 8.]
print(mean_absolute_error(y, p))
print(mean_squared_error(y, p))
print(root_mean_squared_error(y, p))
print(round(r2_score(y, p), 4))`,
  },
  {
    id: 'prf', cat: 'eval', label: 'precision_recall_fscore_support',
    code: `import numpy as np
from sklearn.metrics import fbeta_score, precision_recall_fscore_support

y = [0, 1, 2, 0, 1, 2, 0]
p = [0, 2, 2, 0, 0, 2, 1]
print(precision_recall_fscore_support(y, p, zero_division=0))
print(precision_recall_fscore_support(y, p, average='macro', zero_division=0))
print(fbeta_score([1, 0, 1, 1], [1, 0, 0, 1], beta=2))`,
  },
  {
    id: 'silhouette', cat: 'eval', label: 'silhouette_score',
    code: `import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X = np.array([[0., 0.], [0., 1.], [1., 0.], [9., 9.], [9., 8.], [8., 9.], [0., 9.], [1., 9.]])
for k in [2, 3, 4]:
    labels = KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(X)
    print(k, round(silhouette_score(X, labels), 3))`,
  },
  {
    id: 'splitters', cat: 'eval', label: 'StratifiedKFold / LeaveOneOut',
    code: `import numpy as np
from sklearn.model_selection import LeaveOneOut, ShuffleSplit, StratifiedKFold

X = np.arange(12).reshape(6, 2)
y = np.array([0, 0, 0, 1, 1, 1])
for train, test in StratifiedKFold(3).split(X, y):
    print(train, test, y[test])
print(LeaveOneOut().get_n_splits(X), [t.tolist() for _, t in LeaveOneOut().split(X)][:2])
print([len(t) for _, t in ShuffleSplit(n_splits=2, test_size=2, random_state=0).split(X)])`,
  },
  {
    id: 'column_transformer', cat: 'prep', label: 'ColumnTransformer',
    code: `import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pandas as pd

df = pd.DataFrame({'age': [20, 30, 40], 'city': ['HK', 'SZ', 'HK']})
ct = ColumnTransformer([('num', MinMaxScaler(), ['age']), ('cat', OneHotEncoder(sparse_output=False), ['city'])])
print(ct.fit_transform(df))
print(ct.get_feature_names_out())`,
  },
  {
    id: 'more_prep', cat: 'prep', label: 'OrdinalEncoder / Normalizer',
    code: `import numpy as np
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, Normalizer, OrdinalEncoder, StandardScaler

print(OrdinalEncoder().fit_transform([['low'], ['high'], ['mid']]).ravel())
print(np.round(Normalizer().fit_transform([[3., 4.], [1., 0.]]), 2))
print(FunctionTransformer(np.log1p).fit_transform(np.array([[0.], [1.]])).round(3).ravel())
X = np.array([[1.], [2.], [3.]])
print(np.round(StandardScaler().fit_transform(X).ravel(), 3), MinMaxScaler().fit_transform(X).ravel())
s = StandardScaler().fit(X)
print(s.mean_, np.round(s.scale_, 3), s.inverse_transform([[0.]]))`,
  },
]

export const SKLEARN_PLAYGROUND: PlayConfig = {
  id: 'sklearn',
  libs: [SKLEARN, MATPLOTLIB],
  maxSize: 2_000_000,
  timeBudget: 3000,
  debounce: 400,
  cats: ['data', 'models', 'prep', 'eval'],
  entries: ENTRIES,
  prelude: '',
  callApis: [],
}
