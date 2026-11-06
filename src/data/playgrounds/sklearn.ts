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
]

export const SKLEARN_PLAYGROUND: PlayConfig = {
  id: 'sklearn',
  libs: [SKLEARN, MATPLOTLIB],
  maxSize: 2_000_000,
  timeBudget: 3000,
  debounce: 400,
  cats: ['models', 'prep', 'eval'],
  entries: ENTRIES,
  prelude: '',
  callApis: [],
}
