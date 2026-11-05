import { describe, expect, it } from 'vitest'
import { runPython } from './minipy'
import { SKLEARN } from './pySklearn'
import { PANDAS } from './pyPandas'

const run = (src: string) => runPython(src, { libs: [SKLEARN, PANDAS], maxSize: 200_000, timeBudget: 20_000 })

// Expected stdout and exception from scikit-learn 1.7.2, NumPy 2.2 and pandas 2.3 running the same source. Results that go through
// BLAS, LAPACK or OpenMP are rounded in the source, since their last digits vary between machines (inertia_ sums over threads).
// LogisticRegression is rounded to 3 decimals: the real solver stops at its tolerance, this one fits to convergence.
const CASES: Record<string, { src: string; stdout: string; error?: string; message?: string }> = {
  "split_seeded_same_order": {
    "src": "import numpy as np\nfrom sklearn.model_selection import train_test_split\nX = np.arange(20).reshape(10, 2)\ny = np.arange(10)\nA, B, c, d = train_test_split(X, y, test_size=0.3, random_state=42)\nprint(A)\nprint(B)\nprint(c, d)\nprint(train_test_split(list(range(8)), random_state=3))\nprint(train_test_split(list(range(8)), test_size=3, train_size=2, random_state=0))\nprint(train_test_split(list(range(6)), train_size=0.5, shuffle=False))\nprint(train_test_split(['a', 'b', 'c', 'd', 'e'], [1, 2, 3, 4, 5], random_state=1))",
    "stdout": "[[ 0  1]\n [14 15]\n [ 4  5]\n [18 19]\n [ 8  9]\n [ 6  7]\n [12 13]]\n[[16 17]\n [ 2  3]\n [10 11]]\n[0 7 2 9 4 3 6] [8 1 5]\n[[4, 6, 3, 1, 0, 2], [5, 7]]\n[[7, 3], [6, 2, 1]]\n[[0, 1, 2], [3, 4, 5]]\n[['e', 'a', 'd'], ['c', 'b'], [5, 1, 4], [3, 2]]\n"
  },
  "split_stratified": {
    "src": "import numpy as np\nfrom sklearn.model_selection import train_test_split\ny = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])\na, b, c, d = train_test_split(np.arange(12), y, test_size=0.25, stratify=y, random_state=7)\nprint(a, b, c, d)\nprint(train_test_split(np.arange(10), test_size=0.4, stratify=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1], random_state=0))",
    "stdout": "[ 1  0  4  7  2  5 11  9  8] [ 6  3 10] [0 0 1 1 0 1 2 2 2] [1 0 2]\n[array([3, 2, 1, 9, 6, 5]), array([4, 0, 7, 8])]\n"
  },
  "split_pandas": {
    "src": "import numpy as np\nimport pandas as pd\ndf = pd.DataFrame({'age': [20, 30, 40, 50], 'city': ['a', 'b', 'a', 'c'], 'pay': [1.5, 5.25, 3.5, 9.5], 'z': [1, 2, 3, 4]})\nfrom sklearn.model_selection import train_test_split\na, b = train_test_split(df, test_size=0.5, random_state=1)\nprint(a)\nprint(b)\nc, d = train_test_split(df['pay'], test_size=1, random_state=1)\nprint(c)",
    "stdout": "   age city   pay  z\n0   20    a  1.50  1\n1   30    b  5.25  2\n   age city  pay  z\n3   50    c  9.5  4\n2   40    a  3.5  3\n2    3.50\n0    1.50\n1    5.25\nName: pay, dtype: float64\n"
  },
  "split_errors_consistent_length": {
    "src": "import numpy as np\nfrom sklearn.model_selection import train_test_split\ntrain_test_split([1, 2, 3], [1, 2])",
    "stdout": "",
    "error": "ValueError",
    "message": "Found input variables with inconsistent numbers of samples: [3, 2]"
  },
  "split_errors_too_big": {
    "src": "import numpy as np\nfrom sklearn.model_selection import train_test_split\ntrain_test_split([1, 2, 3], test_size=5)",
    "stdout": "",
    "error": "ValueError",
    "message": "test_size=5 should be either positive and smaller than the number of samples 3 or a float in the (0, 1) range"
  },
  "split_errors_bad_float": {
    "src": "import numpy as np\nfrom sklearn.model_selection import train_test_split\ntrain_test_split([1, 2, 3], test_size=1.5)",
    "stdout": "",
    "error": "InvalidParameterError",
    "message": "The 'test_size' parameter of train_test_split must be a float in the range (0.0, 1.0), an int in the range [1, inf) or None. Got 1.5 instead."
  },
  "split_errors_none": {
    "src": "import numpy as np\nfrom sklearn.model_selection import train_test_split\ntrain_test_split()",
    "stdout": "",
    "error": "ValueError",
    "message": "At least one array required as input"
  },
  "split_errors_stratify_one": {
    "src": "import numpy as np\nfrom sklearn.model_selection import train_test_split\ntrain_test_split([1, 2, 3, 4], stratify=[0, 0, 0, 1], random_state=0)",
    "stdout": "",
    "error": "ValueError",
    "message": "The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2."
  },
  "kfold": {
    "src": "import numpy as np\nfrom sklearn.model_selection import KFold, StratifiedKFold\nfor a, b in KFold(n_splits=3).split(np.arange(7)):\n    print(a, b)\nk = KFold(3, shuffle=True, random_state=0)\nprint(k, k.get_n_splits())\nfor a, b in k.split(np.arange(7)):\n    print(a, b)\ny = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]\nfor a, b in StratifiedKFold(n_splits=2).split(np.zeros((10, 1)), y):\n    print(a, b)\ns = StratifiedKFold(2, shuffle=True, random_state=1)\nprint(s)\nfor a, b in s.split(np.zeros((10, 1)), y):\n    print(a, b)\nprint(next(KFold(2).split(np.arange(4))), len(list(KFold(2).split(np.arange(4)))))",
    "stdout": "[3 4 5 6] [0 1 2]\n[0 1 2 5 6] [3 4]\n[0 1 2 3 4] [5 6]\nKFold(n_splits=3, random_state=0, shuffle=True) 3\n[0 3 4 5] [1 2 6]\n[1 2 4 5 6] [0 3]\n[0 1 2 3 6] [4 5]\n[2 3 7 8 9] [0 1 4 5 6]\n[0 1 4 5 6] [2 3 7 8 9]\nStratifiedKFold(n_splits=2, random_state=1, shuffle=True)\n[0 1 4 7 8] [2 3 5 6 9]\n[2 3 5 6 9] [0 1 4 7 8]\n(array([2, 3]), array([0, 1])) 2\n"
  },
  "kfold_error_one_split": {
    "src": "import numpy as np\nfrom sklearn.model_selection import KFold\nKFold(n_splits=1)",
    "stdout": "",
    "error": "ValueError",
    "message": "k-fold cross-validation requires at least one train/test split by setting n_splits=2 or more, got n_splits=1."
  },
  "kfold_error_seed_without_shuffle": {
    "src": "import numpy as np\nfrom sklearn.model_selection import KFold\nKFold(random_state=1)",
    "stdout": "",
    "error": "ValueError",
    "message": "Setting a random_state has no effect since shuffle is False. You should leave random_state to its default (None), or set shuffle=True."
  },
  "kfold_error_too_many": {
    "src": "import numpy as np\nfrom sklearn.model_selection import KFold\nfor a in KFold(5).split(np.zeros(3)):\n    print(a)",
    "stdout": "",
    "error": "ValueError",
    "message": "Cannot have number of splits n_splits=5 greater than the number of samples: n_samples=3."
  },
  "cross_val": {
    "src": "import numpy as np\nfrom sklearn.model_selection import cross_val_score, KFold, StratifiedKFold\nfrom sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.datasets import load_iris\nX, y = load_iris(return_X_y=True)\nprint(cross_val_score(KNeighborsClassifier(3), X, y))\nprint(cross_val_score(KNeighborsClassifier(3), X, y, cv=3, scoring='accuracy'))\nprint(cross_val_score(KNeighborsClassifier(3), X, y, cv=KFold(3, shuffle=True, random_state=0)))\nprint(cross_val_score(KNeighborsClassifier(3), X, y, cv=StratifiedKFold(4), scoring='f1_macro'))\nA = np.arange(20.0).reshape(-1, 2)\nb = np.array([1.0, 3, 2, 5, 4, 7, 6, 9, 8, 10])\nprint(np.round(cross_val_score(LinearRegression(), A, b, cv=2), 6))\nprint(np.round(cross_val_score(LinearRegression(), A, b, cv=2, scoring='neg_mean_squared_error'), 6))",
    "stdout": "[0.96666667 0.96666667 0.93333333 0.96666667 1.        ]\n[0.98 0.96 0.98]\n[0.96 0.94 0.96]\n[0.97333333 0.94588745 0.92       1.        ]\n[0.14 0.14]\n[-1.72 -1.72]\n"
  },
  "grid_search": {
    "src": "import numpy as np\nfrom sklearn.model_selection import GridSearchCV\nfrom sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.datasets import load_iris\nX, y = load_iris(return_X_y=True)\ng = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': [1, 3, 5], 'weights': ['uniform', 'distance']}, cv=3)\ng.fit(X, y)\nprint(g.best_params_, g.best_score_, g.best_index_, g.best_estimator_)\nprint(g.cv_results_['mean_test_score'], g.cv_results_['rank_test_score'], g.cv_results_['params'][0])\nprint(g.predict(X[:3]), g.score(X, y))\nprint(g)\np = GridSearchCV(make_pipeline(StandardScaler(), KNeighborsClassifier()), {'kneighborsclassifier__n_neighbors': [1, 5]}, scoring='accuracy')\nprint(p.fit(X, y).best_params_, p.best_score_)",
    "stdout": "{'n_neighbors': 5, 'weights': 'uniform'} 0.98 4 KNeighborsClassifier()\n[0.96       0.96       0.97333333 0.97333333 0.98       0.97333333] [5 5 2 2 1 2] {'n_neighbors': 1, 'weights': 'uniform'}\n[0 0 0] 0.9666666666666667\nGridSearchCV(cv=3, estimator=KNeighborsClassifier(),\n             param_grid={'n_neighbors': [1, 3, 5],\n                         'weights': ['uniform', 'distance']})\n{'kneighborsclassifier__n_neighbors': 5} 0.96\n"
  },
  "label_encoder": {
    "src": "import numpy as np\nfrom sklearn.preprocessing import LabelEncoder\ne = LabelEncoder()\nprint(e.fit_transform(['red', 'green', 'red', 'blue']), e.classes_, repr(e.classes_))\nprint(e.transform(['blue']), e.inverse_transform([0, 2]), repr(e.inverse_transform([1])), e.inverse_transform([0, 2]).tolist())\nn = LabelEncoder().fit([10, 5, 10])\nprint(n.classes_, repr(n.classes_), n.transform([10, 5]), n.inverse_transform([1, 0]))\nprint(e)",
    "stdout": "[2 1 2 0] ['blue' 'green' 'red'] array(['blue', 'green', 'red'], dtype='<U5')\n[0] ['blue' 'red'] array(['green'], dtype='<U5') ['blue', 'red']\n[ 5 10] array([ 5, 10]) [1 0] [10  5]\nLabelEncoder()\n"
  },
  "label_encoder_on_a_frame": {
    "src": "import numpy as np\nimport pandas as pd\ndf = pd.DataFrame({'age': [20, 30, 40, 50], 'city': ['a', 'b', 'a', 'c'], 'pay': [1.5, 5.25, 3.5, 9.5], 'z': [1, 2, 3, 4]})\nfrom sklearn.preprocessing import LabelEncoder\nencoder = LabelEncoder()\ndf['city'] = encoder.fit_transform(df['city'])\nprint(df)\nprint(encoder.classes_, repr(encoder.classes_))",
    "stdout": "   age  city   pay  z\n0   20     0  1.50  1\n1   30     1  5.25  2\n2   40     0  3.50  3\n3   50     2  9.50  4\n['a' 'b' 'c'] array(['a', 'b', 'c'], dtype=object)\n"
  },
  "label_encoder_unseen_text": {
    "src": "import numpy as np\nfrom sklearn.preprocessing import LabelEncoder\nLabelEncoder().fit(['a', 'b']).transform(['z'])",
    "stdout": "",
    "error": "ValueError",
    "message": "y contains previously unseen labels: np.str_('z')"
  },
  "label_encoder_unseen_number": {
    "src": "import numpy as np\nfrom sklearn.preprocessing import LabelEncoder\nLabelEncoder().fit([1, 2]).transform([5])",
    "stdout": "",
    "error": "ValueError",
    "message": "y contains previously unseen labels: [np.int64(5)]"
  },
  "label_encoder_inverse_unseen": {
    "src": "import numpy as np\nfrom sklearn.preprocessing import LabelEncoder\nLabelEncoder().fit([1, 2]).inverse_transform([5])",
    "stdout": "",
    "error": "ValueError",
    "message": "y contains previously unseen labels: [5]"
  },
  "one_hot": {
    "src": "import numpy as np\nrows = [['red', 's'], ['green', 'm'], ['blue', 's'], ['green', 'l'], ['red', 'm']]\nfrom sklearn.preprocessing import OneHotEncoder\ne = OneHotEncoder()\nS = e.fit_transform(rows)\nprint(S.shape, S.nnz)\nprint(repr(S))\nprint(S)\nprint(S.toarray())\nprint(e.categories_)\nprint(e.get_feature_names_out(), repr(e.get_feature_names_out(['c', 'd'])))\nprint(e.inverse_transform(S))\nprint(e)",
    "stdout": "(5, 6) 10\n<Compressed Sparse Row sparse matrix of dtype 'float64'\n\twith 10 stored elements and shape (5, 6)>\n<Compressed Sparse Row sparse matrix of dtype 'float64'\n\twith 10 stored elements and shape (5, 6)>\n  Coords\tValues\n  (0, 2)\t1.0\n  (0, 5)\t1.0\n  (1, 1)\t1.0\n  (1, 4)\t1.0\n  (2, 0)\t1.0\n  (2, 5)\t1.0\n  (3, 1)\t1.0\n  (3, 3)\t1.0\n  (4, 2)\t1.0\n  (4, 4)\t1.0\n[[0. 0. 1. 0. 0. 1.]\n [0. 1. 0. 0. 1. 0.]\n [1. 0. 0. 0. 0. 1.]\n [0. 1. 0. 1. 0. 0.]\n [0. 0. 1. 0. 1. 0.]]\n[array(['blue', 'green', 'red'], dtype=object), array(['l', 'm', 's'], dtype=object)]\n['x0_blue' 'x0_green' 'x0_red' 'x1_l' 'x1_m' 'x1_s'] array(['c_blue', 'c_green', 'c_red', 'd_l', 'd_m', 'd_s'], dtype=object)\n[['red' 's']\n ['green' 'm']\n ['blue' 's']\n ['green' 'l']\n ['red' 'm']]\nOneHotEncoder()\n"
  },
  "one_hot_options": {
    "src": "import numpy as np\nrows = [['red', 's'], ['green', 'm'], ['blue', 's'], ['green', 'l'], ['red', 'm']]\nfrom sklearn.preprocessing import OneHotEncoder\ne = OneHotEncoder(sparse_output=False, drop='first')\nprint(e.fit_transform(rows), e.drop_idx_, e.get_feature_names_out())\nprint(e)\ni = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(rows)\nprint(i.transform([['nope', 's']]))\nb = OneHotEncoder(sparse_output=False, drop='if_binary').fit_transform([['a', 'x'], ['b', 'y'], ['c', 'x']])\nprint(b)\nn = OneHotEncoder(sparse_output=False, dtype=int).fit(np.array([3, 1, 3]).reshape(-1, 1))\nprint(n.transform([[1], [3]]), n.categories_, n.get_feature_names_out(), n.inverse_transform([[1, 0]]))\nc = OneHotEncoder(categories=[['s', 'm', 'l']], sparse_output=False).fit([['m'], ['l']])\nprint(c.transform([['s']]), c.categories_)",
    "stdout": "[[0. 1. 0. 1.]\n [1. 0. 1. 0.]\n [0. 0. 0. 1.]\n [1. 0. 0. 0.]\n [0. 1. 1. 0.]] [0 0] ['x0_green' 'x0_red' 'x1_m' 'x1_s']\nOneHotEncoder(drop='first', sparse_output=False)\n[[0. 0. 0. 0. 0. 1.]]\n[[1. 0. 0. 0.]\n [0. 1. 0. 1.]\n [0. 0. 1. 0.]]\n[[1 0]\n [0 1]] [array([1, 3])] ['x0_1' 'x0_3'] [[1]]\n[[1. 0. 0.]] [array(['s', 'm', 'l'], dtype=object)]\n"
  },
  "one_hot_on_a_frame": {
    "src": "import numpy as np\nimport pandas as pd\ndf = pd.DataFrame({'age': [20, 30, 40, 50], 'city': ['a', 'b', 'a', 'c'], 'pay': [1.5, 5.25, 3.5, 9.5], 'z': [1, 2, 3, 4]})\nfrom sklearn.preprocessing import OneHotEncoder\ne = OneHotEncoder(sparse_output=False)\nprint(e.fit_transform(df[['city']]), e.categories_, e.get_feature_names_out(), e.feature_names_in_)",
    "stdout": "[[1. 0. 0.]\n [0. 1. 0.]\n [1. 0. 0.]\n [0. 0. 1.]] [array(['a', 'b', 'c'], dtype=object)] ['city_a' 'city_b' 'city_c'] ['city']\n"
  },
  "one_hot_needs_2d": {
    "src": "import numpy as np\nfrom sklearn.preprocessing import OneHotEncoder\nOneHotEncoder().fit(['a', 'b'])",
    "stdout": "",
    "error": "ValueError",
    "message": "Expected 2D array, got 1D array instead:\narray=['a' 'b'].\nReshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample."
  },
  "one_hot_unknown_text": {
    "src": "import numpy as np\nfrom sklearn.preprocessing import OneHotEncoder\nOneHotEncoder().fit([['a'], ['b']]).transform([['z']])",
    "stdout": "",
    "error": "ValueError",
    "message": "Found unknown categories ['z'] in column 0 during transform"
  },
  "one_hot_unknown_number": {
    "src": "import numpy as np\nfrom sklearn.preprocessing import OneHotEncoder\nOneHotEncoder().fit([[1], [2]]).transform([[9]])",
    "stdout": "",
    "error": "ValueError",
    "message": "Found unknown categories [np.int64(9)] in column 0 during transform"
  },
  "ordinal": {
    "src": "import numpy as np\nrows = [['red', 's'], ['green', 'm'], ['blue', 's'], ['green', 'l'], ['red', 'm']]\nfrom sklearn.preprocessing import OrdinalEncoder\ne = OrdinalEncoder()\nprint(e.fit_transform(rows), e.categories_, e.inverse_transform([[0, 0]]))\nprint(e)\nu = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit(rows)\nprint(u.transform([['nope', 's']]), u)",
    "stdout": "[[2. 2.]\n [1. 1.]\n [0. 2.]\n [1. 0.]\n [2. 1.]] [array(['blue', 'green', 'red'], dtype=object), array(['l', 'm', 's'], dtype=object)] [['blue' 'l']]\nOrdinalEncoder()\n[[-1.  2.]] OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)\n"
  },
  "ordinal_unknown": {
    "src": "import numpy as np\nfrom sklearn.preprocessing import OrdinalEncoder\nOrdinalEncoder().fit([['a'], ['b']]).transform([['z']])",
    "stdout": "",
    "error": "ValueError",
    "message": "Found unknown categories ['z'] in column 0 during transform"
  },
  "standard_scaler": {
    "src": "import numpy as np\nfrom sklearn.preprocessing import StandardScaler\nX = np.array([[0, 1.5], [3, 2.5], [2, -1], [0, 7], [5, 0.25]])\ns = StandardScaler()\nZ = s.fit_transform(X)\nprint(Z.tolist())\nprint(s.mean_.tolist(), s.var_.tolist(), s.scale_.tolist(), s.n_samples_seen_, s.n_features_in_)\nprint(s.inverse_transform(s.transform(X[:1])))\nprint(StandardScaler().fit_transform(np.array([[0], [3], [2], [0], [5], [3], [4], [4], [5], [3]])).tolist())\nprint(StandardScaler().fit_transform(np.array([[1.0, 5], [1, 7], [1, 9]])))\nw = StandardScaler(with_mean=False).fit([[1.0, 2], [3, 4]])\nprint(w, w.mean_, w.scale_, w.transform([[1.0, 2]]))\nt = StandardScaler(with_std=False).fit([[1.0, 2], [3, 4]])\nprint(t.mean_, t.var_, t.scale_, t.transform([[1.0, 2]]))",
    "stdout": "[[-1.0540925533894598, -0.2006978504214273], [0.5270462766947299, 0.16420733216298608], [0.0, -1.1129608068824608], [-1.0540925533894598, 1.8062806537928462], [1.5811388300841898, -0.656829328651944]]\n[2.0, 2.05] [3.6, 7.510000000000001] [1.8973665961010275, 2.7404379212089442] 5 2\n[[0.  1.5]]\n[[-1.7058823529411764], [0.05882352941176476], [-0.5294117647058824], [-1.7058823529411764], [1.2352941176470589], [0.05882352941176476], [0.6470588235294118], [0.6470588235294118], [1.2352941176470589], [0.05882352941176476]]\n[[ 0.         -1.22474487]\n [ 0.          0.        ]\n [ 0.          1.22474487]]\nStandardScaler(with_mean=False) [2. 3.] [1. 1.] [[1. 2.]]\n[2. 3.] None None [[-1. -1.]]\n"
  },
  "scaler_needs_2d": {
    "src": "import numpy as np\nfrom sklearn.preprocessing import StandardScaler\nStandardScaler().fit([1, 2, 3])",
    "stdout": "",
    "error": "ValueError",
    "message": "Expected 2D array, got 1D array instead:\narray=[1. 2. 3.].\nReshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample."
  },
  "scaler_not_fitted": {
    "src": "import numpy as np\nfrom sklearn.preprocessing import StandardScaler\nStandardScaler().transform([[1, 2]])",
    "stdout": "",
    "error": "NotFittedError",
    "message": "This StandardScaler instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
  },
  "minmax": {
    "src": "import numpy as np\nfrom sklearn.preprocessing import MinMaxScaler\nX = np.array([[1, 2.0], [3, 4.5], [5, 4.0], [7, 4.5]])\ns = MinMaxScaler()\nprint(s.fit_transform(X))\nprint(s.data_min_, s.data_max_, s.data_range_, s.scale_, s.min_)\nr = MinMaxScaler(feature_range=(-1, 2)).fit(X)\nprint(r.transform(X).tolist(), r.inverse_transform(r.transform(X[:1])), r)\nprint(MinMaxScaler(clip=True).fit(X).transform([[100, -100]]))\nprint(MinMaxScaler().fit_transform([[2, 5], [2, 7]]))",
    "stdout": "[[0.         0.        ]\n [0.33333333 1.        ]\n [0.66666667 0.8       ]\n [1.         1.        ]]\n[1. 2.] [7.  4.5] [6.  2.5] [0.16666667 0.4       ] [-0.16666667 -0.8       ]\n[[-1.0, -1.0], [0.0, 1.9999999999999996], [1.0, 1.4], [2.0, 1.9999999999999996]] [[1. 2.]] MinMaxScaler(feature_range=(-1, 2))\n[[1. 0.]]\n[[0. 0.]\n [0. 1.]]\n"
  },
  "normalizer": {
    "src": "import numpy as np\nfrom sklearn.preprocessing import Normalizer\nX = np.array([[3.0, 4.0], [1.0, 1.0], [0.0, 0.0], [-2.0, 0.5]])\nprint(np.round(Normalizer().fit_transform(X), 12))\nprint(np.round(Normalizer('l1').fit_transform(X), 12))\nprint(np.round(Normalizer(norm='max').transform(X), 12), Normalizer(norm='l1'))",
    "stdout": "[[ 0.6         0.8       ]\n [ 0.70710678  0.70710678]\n [ 0.          0.        ]\n [-0.9701425   0.24253563]]\n[[ 0.42857143  0.57142857]\n [ 0.5         0.5       ]\n [ 0.          0.        ]\n [-0.8         0.2       ]]\n[[ 0.75  1.  ]\n [ 1.    1.  ]\n [ 0.    0.  ]\n [-1.    0.25]] Normalizer(norm='l1')\n"
  },
  "polynomial": {
    "src": "import numpy as np\nfrom sklearn.preprocessing import PolynomialFeatures\nX = np.array([[2.0, 3.0], [0.5, -1.0]])\np = PolynomialFeatures(2)\nprint(p.fit_transform(X), p.get_feature_names_out(), p.powers_, p.n_output_features_, p)\nprint(PolynomialFeatures(3, include_bias=False).fit_transform(X).tolist())\nprint(PolynomialFeatures(2, interaction_only=True).fit_transform([[1.0, 2, 3]]).tolist())\nq = PolynomialFeatures((2, 3)).fit([[2.0, 3.0]])\nprint(q.transform([[2.0, 3.0]]).tolist(), q.get_feature_names_out(['a', 'b']))",
    "stdout": "[[ 1.    2.    3.    4.    6.    9.  ]\n [ 1.    0.5  -1.    0.25 -0.5   1.  ]] ['1' 'x0' 'x1' 'x0^2' 'x0 x1' 'x1^2'] [[0 0]\n [1 0]\n [0 1]\n [2 0]\n [1 1]\n [0 2]] 6 PolynomialFeatures()\n[[2.0, 3.0, 4.0, 6.0, 9.0, 8.0, 12.0, 18.0, 27.0], [0.5, -1.0, 0.25, -0.5, 1.0, 0.125, -0.25, 0.5, -1.0]]\n[[1.0, 1.0, 2.0, 3.0, 2.0, 3.0, 6.0]]\n[[1.0, 4.0, 6.0, 9.0, 8.0, 12.0, 18.0, 27.0]] ['1' 'a^2' 'a b' 'b^2' 'a^3' 'a^2 b' 'a b^2' 'b^3']\n"
  },
  "function_transformer": {
    "src": "import numpy as np\nfrom sklearn.preprocessing import FunctionTransformer\nX = np.array([[0.0, 1], [2, 3]])\nf = FunctionTransformer(np.log1p, inverse_func=np.expm1)\nprint(np.round(f.fit_transform(X), 6), np.round(f.inverse_transform(f.transform(X)), 6))\ng = FunctionTransformer()\nprint(g.fit_transform(X), g)\nh = FunctionTransformer(lambda x: x * 2)\nprint(h.fit_transform([[1, 2]]))\ndef scale(x, by=1):\n    return x * by\nk = FunctionTransformer(scale, kw_args={'by': 3})\nprint(k.fit_transform(X))\no = FunctionTransformer(np.sqrt, feature_names_out='one-to-one').fit(X)\nprint(o.get_feature_names_out(), o.transform(X).shape)",
    "stdout": "[[0.       0.693147]\n [1.098612 1.386294]] [[0. 1.]\n [2. 3.]]\n[[0. 1.]\n [2. 3.]] FunctionTransformer()\n[[1, 2], [1, 2]]\n[[0. 3.]\n [6. 9.]]\n['x0' 'x1'] (2, 2)\n"
  },
  "pipeline": {
    "src": "import numpy as np\nX = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [1.0, 0.6], [9.0, 11.0], [8.0, 2.0], [10.0, 2.0], [9.0, 3.0]])\ny = [0, 0, 1, 1, 0, 1, 2, 2, 2]\nfrom sklearn.pipeline import Pipeline, make_pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.neighbors import KNeighborsClassifier\np = Pipeline([('s', StandardScaler()), ('k', KNeighborsClassifier(3))])\nprint(p)\nprint(p.fit(X, y).predict(X), p.score(X, y), p.predict_proba(X[:1]), p.classes_)\nprint(p.named_steps['k'], p['s'], p[0], len(p), p.named_steps.k)\nprint(p.steps)\nprint(p[:1])\nq = make_pipeline(StandardScaler(), KNeighborsClassifier(1))\nprint(q)\nprint(q.steps[0][0], q.fit(X, y).score(X, y))\nprint(p.get_params()['k__n_neighbors'], p.set_params(k__n_neighbors=1).fit(X, y).predict(X))\nprint(make_pipeline(StandardScaler(), StandardScaler()).steps[1][0])",
    "stdout": "Pipeline(steps=[('s', StandardScaler()),\n                ('k', KNeighborsClassifier(n_neighbors=3))])\n[0 0 1 1 0 1 2 2 2] 1.0 [[1. 0. 0.]] [0 1 2]\nKNeighborsClassifier(n_neighbors=3) StandardScaler() StandardScaler() 2 KNeighborsClassifier(n_neighbors=3)\n[('s', StandardScaler()), ('k', KNeighborsClassifier(n_neighbors=3))]\nPipeline(steps=[('s', StandardScaler())])\nPipeline(steps=[('standardscaler', StandardScaler()),\n                ('kneighborsclassifier', KNeighborsClassifier(n_neighbors=1))])\nstandardscaler 1.0\n3 [0 0 1 1 0 1 2 2 2]\nstandardscaler-2\n"
  },
  "pipeline_transform": {
    "src": "import numpy as np\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler, PolynomialFeatures\nfrom sklearn.linear_model import LinearRegression\nX = np.array([[0.0], [1], [2], [3], [4]])\ny = [1.0, 2.5, 5.2, 9.9, 17.1]\np = make_pipeline(PolynomialFeatures(2), LinearRegression())\nprint(p)\np.fit(X, y)\nprint(np.round(p.predict(X), 6), round(p.score(X, y), 6))\nprint(np.round(p[1].coef_, 6))\nt = make_pipeline(StandardScaler(), PolynomialFeatures(2))\nprint(np.round(t.fit_transform(X), 6), np.round(t.transform(X[:1]), 6))",
    "stdout": "Pipeline(steps=[('polynomialfeatures', PolynomialFeatures()),\n                ('linearregression', LinearRegression())])\n[ 1.134286  2.222857  5.225714 10.142857 16.974286] 0.998997\n[0.       0.131429 0.957143]\n[[ 1.       -1.414214  2.      ]\n [ 1.       -0.707107  0.5     ]\n [ 1.        0.        0.      ]\n [ 1.        0.707107  0.5     ]\n [ 1.        1.414214  2.      ]] [[ 1.       -1.414214  2.      ]]\n"
  },
  "pipeline_needs_transformers": {
    "src": "import numpy as np\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.preprocessing import StandardScaler\nmake_pipeline(KNeighborsClassifier(), StandardScaler()).fit([[1.0], [2.0]], [0, 1])",
    "stdout": "",
    "error": "TypeError"
  },
  "column_transformer_lab": {
    "src": "import numpy as np\nimport pandas as pd\ndf = pd.DataFrame({'age': [20, 30, 40, 50], 'city': ['a', 'b', 'a', 'c'], 'pay': [1.5, 5.25, 3.5, 9.5], 'z': [1, 2, 3, 4]})\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer\nfrom sklearn.pipeline import Pipeline\nnum_cols = ['age', 'pay']\ncat_cols = ['city']\nct = ColumnTransformer([('num', Pipeline([('log', FunctionTransformer(lambda x: np.log10(x + 1))), ('scaler', StandardScaler())]), num_cols), ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)])\nZ = ct.fit_transform(df)\nprint(np.round(Z, 6))\nprint(ct.output_indices_)\nprint(np.round(ct.transform(df.head(2)), 6))\nprint(ct.named_transformers_['cat'].categories_)\nprint(ct.transformers_[1][0], ct.transformers_[2])\nprint(ct.n_features_in_, ct.feature_names_in_)",
    "stdout": "[[-1.466606 -1.41092   1.        0.        0.      ]\n [-0.292413  0.348511  0.        1.        0.      ]\n [ 0.550504 -0.282272  1.        0.        0.      ]\n [ 1.208514  1.344681  0.        0.        1.      ]]\n{'num': slice(0, 2, None), 'cat': slice(2, 5, None), 'remainder': slice(0, 0, None)}\n[[-1.466606 -1.41092   1.        0.        0.      ]\n [-0.292413  0.348511  0.        1.        0.      ]]\n[array(['a', 'b', 'c'], dtype=object)]\ncat ('remainder', 'drop', ['z'])\n4 ['age' 'city' 'pay' 'z']\n"
  },
  "column_transformer_remainder": {
    "src": "import numpy as np\nimport pandas as pd\ndf = pd.DataFrame({'age': [20, 30, 40, 50], 'city': ['a', 'b', 'a', 'c'], 'pay': [1.5, 5.25, 3.5, 9.5], 'z': [1, 2, 3, 4]})\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.preprocessing import OneHotEncoder, StandardScaler\ndf2 = df[['age', 'city', 'pay']]\nct = ColumnTransformer([('n', StandardScaler(), ['age']), ('c', OneHotEncoder(), ['city'])], remainder='passthrough')\nprint(np.round(ct.fit_transform(df2), 6))\nprint(ct.get_feature_names_out())\nprint(ct)\nd = ColumnTransformer([('c', OneHotEncoder(), ['city'])])\nS = d.fit_transform(df2)\nprint(S.shape, np.round(S, 3))\nprint(d.get_feature_names_out())",
    "stdout": "[[-1.341641  1.        0.        0.        1.5     ]\n [-0.447214  0.        1.        0.        5.25    ]\n [ 0.447214  1.        0.        0.        3.5     ]\n [ 1.341641  0.        0.        1.        9.5     ]]\n['n__age' 'c__city_a' 'c__city_b' 'c__city_c' 'remainder__pay']\nColumnTransformer(remainder='passthrough',\n                  transformers=[('n', StandardScaler(), ['age']),\n                                ('c', OneHotEncoder(), ['city'])])\n(4, 3) [[1. 0. 0.]\n [0. 1. 0.]\n [1. 0. 0.]\n [0. 0. 1.]]\n['c__city_a' 'c__city_b' 'c__city_c']\n"
  },
  "column_transformer_array": {
    "src": "import numpy as np\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder\nX = np.array([[1.0, 2, 3], [4, 5, 6], [7, 8, 10]])\nct = ColumnTransformer([('a', StandardScaler(), [0, 1]), ('b', MinMaxScaler(), [2])])\nprint(np.round(ct.fit_transform(X), 6), ct.get_feature_names_out())\nprint(ct)\nd = ColumnTransformer([('a', 'passthrough', [0]), ('b', 'drop', [1])], remainder='passthrough')\nprint(d.fit_transform(X))\ns = ColumnTransformer([('o', OneHotEncoder(), [0])])\nprint(s.fit_transform(np.array([[1.0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]])))",
    "stdout": "[[-1.224745 -1.224745  0.      ]\n [ 0.        0.        0.428571]\n [ 1.224745  1.224745  1.      ]] ['a__x0' 'a__x1' 'b__x2']\nColumnTransformer(transformers=[('a', StandardScaler(), [0, 1]),\n                                ('b', MinMaxScaler(), [2])])\n[[ 1.  3.]\n [ 4.  6.]\n [ 7. 10.]]\n<Compressed Sparse Row sparse matrix of dtype 'float64'\n\twith 6 stored elements and shape (6, 5)>\n  Coords\tValues\n  (0, 0)\t1.0\n  (1, 0)\t1.0\n  (2, 1)\t1.0\n  (3, 2)\t1.0\n  (4, 3)\t1.0\n  (5, 4)\t1.0\n"
  },
  "column_transformer_names_need_a_frame": {
    "src": "import numpy as np\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.preprocessing import StandardScaler\nColumnTransformer([('a', StandardScaler(), ['x'])]).fit(np.zeros((2, 2)))",
    "stdout": "",
    "error": "ValueError",
    "message": "Specifying the columns using strings is only supported for dataframes."
  },
  "knn_classifier": {
    "src": "import numpy as np\nX = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [1.0, 0.6], [9.0, 11.0], [8.0, 2.0], [10.0, 2.0], [9.0, 3.0]])\ny = [0, 0, 1, 1, 0, 1, 2, 2, 2]\nfrom sklearn.neighbors import KNeighborsClassifier\nq = np.array([[1.0, 1.0], [9.0, 9.0], [9.0, 2.5], [5.0, 5.0]])\nm = KNeighborsClassifier(n_neighbors=3)\nprint(m)\nprint(m.fit(X, y).predict(q), m.classes_, m.score(q, [0, 1, 2, 0]))\nprint(np.round(m.predict_proba(q), 6))\nd, i = m.kneighbors(q)\nprint(np.round(d, 6), i)\nprint(m.kneighbors(q[:1], n_neighbors=1, return_distance=False))\nd2, i2 = m.kneighbors()\nprint(np.round(d2[:2], 6), i2[:2])\nprint(m.n_samples_fit_, m.effective_metric_, m.n_features_in_)\nfor kw in [{'weights': 'distance'}, {'metric': 'manhattan'}, {'metric': 'minkowski', 'p': 3}, {'p': 1}, {'metric': 'euclidean'}, {'metric': 'chebyshev'}, {'algorithm': 'brute'}]:\n    n = KNeighborsClassifier(3, **kw).fit(X, y)\n    print(n, n.predict(q), np.round(n.predict_proba(q)[3], 6), np.round(n.kneighbors(q[:1])[0], 6))",
    "stdout": "KNeighborsClassifier(n_neighbors=3)\n[0 1 2 1] [0 1 2] 0.75\n[[1.       0.       0.      ]\n [0.       1.       0.      ]\n [0.       0.       1.      ]\n [0.       0.666667 0.333333]]\n[[0.4      0.943398 1.      ]\n [1.414214 2.       4.123106]\n [0.5      1.118034 1.118034]\n [3.       4.242641 4.242641]] [[4 1 0]\n [3 5 2]\n [8 6 7]\n [2 6 3]]\n[[4]]\n[[0.538516 1.4      7.      ]\n [0.538516 1.3      6.503076]] [[1 4 6]\n [0 4 6]]\n9 euclidean 2\nKNeighborsClassifier(n_neighbors=3, weights='distance') [0 1 2 1] [0.       0.707107 0.292893] [[0.4      0.943398 1.      ]]\nKNeighborsClassifier(metric='manhattan', n_neighbors=3) [0 1 2 1] [0.       0.666667 0.333333] [[0.4 1.  1.3]]\nKNeighborsClassifier(n_neighbors=3, p=3) [0 1 2 1] [0.       0.666667 0.333333] [[0.4      0.860425 1.      ]]\nKNeighborsClassifier(n_neighbors=3, p=1) [0 1 2 1] [0.       0.666667 0.333333] [[0.4 1.  1.3]]\nKNeighborsClassifier(metric='euclidean', n_neighbors=3) [0 1 2 1] [0.       0.666667 0.333333] [[0.4      0.943398 1.      ]]\nKNeighborsClassifier(metric='chebyshev', n_neighbors=3) [0 1 2 1] [0.       0.666667 0.333333] [[0.4 0.8 1. ]]\nKNeighborsClassifier(algorithm='brute', n_neighbors=3) [0 1 2 1] [0.       0.666667 0.333333] [[0.4      0.943398 1.      ]]\n"
  },
  "knn_exact_point_takes_all_weight": {
    "src": "import numpy as np\nfrom sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor\nX = [[0.0], [1.0], [2.0], [3.0]]\nm = KNeighborsClassifier(3, weights='distance').fit(X, [0, 1, 1, 0])\nprint(m.predict([[1.0]]), m.predict_proba([[1.0]]))\nr = KNeighborsRegressor(3, weights='distance').fit(X, [1.0, 2.0, 3.0, 4.0])\nprint(r.predict([[1.0], [1.5]]))",
    "stdout": "[1] [[0. 1.]]\n[2.         2.28571429]\n"
  },
  "knn_text_labels": {
    "src": "import numpy as np\nX = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [1.0, 0.6], [9.0, 11.0], [8.0, 2.0], [10.0, 2.0], [9.0, 3.0]])\ny = [0, 0, 1, 1, 0, 1, 2, 2, 2]\nfrom sklearn.neighbors import KNeighborsClassifier\nt = ['a', 'a', 'b', 'b', 'a', 'b', 'c', 'c', 'c']\nm = KNeighborsClassifier(3).fit(X, t)\np = m.predict(X[:3])\nprint(m.classes_, repr(m.classes_), p, repr(p), p == np.array([1, 1, 1]) if False else p[0], m.score(X, t))\nprint(p.shape, len(p), p.tolist(), (p == t[:3]).sum(), p[1:], p[p == 'a'])",
    "stdout": "['a' 'b' 'c'] array(['a', 'b', 'c'], dtype='<U1') ['a' 'a' 'b'] array(['a', 'a', 'b'], dtype='<U1') a 1.0\n(3,) 3 ['a', 'a', 'b'] 3 ['a' 'b'] ['a' 'a']\n"
  },
  "knn_ties_float": {
    "src": "import numpy as np\nfrom sklearn.neighbors import KNeighborsClassifier\nX = np.array([[0.0, 0], [2, 1], [0, 1], [5, 0], [0, 4], [1, 4], [2, 0], [0, 5], [5, 4], [4, 5], [3, 5], [2, 1], [1, 3], [0, 0], [2, 2], [3, 3], [4, 0], [1, 1], [5, 5], [3, 0], [1, 5], [4, 4]])\ny = np.arange(22) % 3\nq = np.array([[1.0, 2.0], [3.0, 1.0], [2.0, 3.0]])\nfor k in [1, 2, 3, 4, 5, 11, 12, 15]:\n    m = KNeighborsClassifier(k).fit(X, y)\n    print(k, m._fit_method, m.kneighbors(q, return_distance=False).tolist(), m.predict(q))",
    "stdout": "1 kd_tree [[12], [1], [12]] [0 1 0]\n2 kd_tree [[14, 12], [11, 1], [14, 12]] [0 1 0]\n3 kd_tree [[17, 12, 14], [19, 11, 1], [15, 14, 12]] [2 1 0]\n4 kd_tree [[17, 14, 12, 2], [19, 11, 1, 6], [15, 14, 12, 5]] [2 1 0]\n5 kd_tree [[14, 17, 12, 1, 11], [1, 19, 11, 6, 14], [12, 15, 14, 5, 1]] [2 1 0]\n11 brute [[17, 12, 14, 1, 11, 2, 5, 6, 4, 13, 0], [11, 1, 19, 6, 14, 16, 15, 17, 3, 12, 2], [14, 15, 12, 5, 11, 1, 4, 20, 10, 17, 21]] [2 0 2]\n12 brute [[17, 12, 14, 2, 1, 11, 5, 13, 0, 15, 4, 6], [1, 19, 11, 6, 14, 16, 15, 17, 3, 12, 2, 13], [15, 12, 14, 5, 1, 11, 21, 20, 10, 4, 17, 9]] [2 0 2]\n15 brute [[12, 17, 14, 11, 1, 2, 5, 15, 4, 0, 6, 13, 19, 20, 7], [11, 19, 1, 14, 6, 16, 17, 15, 3, 12, 2, 21, 13, 0, 8], [15, 14, 12, 5, 11, 1, 10, 4, 20, 21, 17, 2, 9, 7, 6]] [2 0 2]\n"
  },
  "knn_ties_int": {
    "src": "import numpy as np\nfrom sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor\nX = np.array([[0, 0], [2, 1], [0, 1], [5, 0], [0, 4], [1, 4], [2, 0], [0, 5], [5, 4], [4, 5], [3, 5], [2, 1], [1, 3], [0, 0], [2, 2]])\ny = [1, 0, 1, 0, 1, 0, 0, 2, 2, 0, 0, 0, 1, 1, 0]\nq = np.array([[3, 4], [5, 1], [1, 1]])\nfor k in [1, 3, 5, 7, 9, 14]:\n    m = KNeighborsClassifier(k).fit(X, y)\n    print(k, m.kneighbors(q, return_distance=False).tolist(), m.predict(q), m.kneighbors(q[:1], return_distance=False).tolist())\nd, i = KNeighborsClassifier(7).fit(X, y).kneighbors()\nprint(i[:3].tolist())\nprint(KNeighborsRegressor(9).fit(X, y).predict(q))",
    "stdout": "1 [[10], [3], [1]] [0 0 0] [[10]]\n3 [[10, 9, 5], [3, 8, 1], [11, 2, 1]] [0 0 0] [[10, 9, 5]]\n5 [[10, 9, 5, 8, 12], [3, 1, 8, 11, 6], [1, 11, 2, 0, 6]] [0 0 0] [[10, 9, 5, 8, 12]]\n7 [[10, 9, 5, 8, 12, 14, 4], [3, 11, 8, 1, 6, 14, 9], [1, 2, 11, 14, 13, 0, 6]] [0 0 0] [[10, 9, 5, 8, 12, 14, 4]]\n9 [[10, 9, 8, 5, 14, 12, 4, 11, 7], [3, 8, 11, 1, 14, 6, 9, 10, 12], [1, 2, 11, 14, 13, 0, 6, 12, 5]] [0 0 0] [[10, 9, 8, 5, 14, 12, 4, 11, 7]]\n14 [[10, 9, 8, 5, 14, 12, 4, 11, 7, 1, 6, 2, 3, 13], [3, 1, 8, 11, 14, 6, 9, 12, 10, 2, 5, 13, 0, 4], [1, 2, 11, 14, 13, 0, 6, 12, 5, 4, 3, 7, 10, 8]] [0 0 0] [[10, 9, 8, 5, 14, 12, 4, 11, 7, 1, 6, 2, 3, 13]]\n[[13, 2, 6, 11, 1, 14, 12], [11, 14, 6, 2, 13, 12, 0], [0, 13, 1, 11, 12, 14, 6]]\n[0.66666667 0.33333333 0.44444444]\n"
  },
  "knn_regressor": {
    "src": "import numpy as np\nX = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [1.0, 0.6], [9.0, 11.0], [8.0, 2.0], [10.0, 2.0], [9.0, 3.0]])\ny = [0, 0, 1, 1, 0, 1, 2, 2, 2]\nfrom sklearn.neighbors import KNeighborsRegressor\nt = [1.5, 2.5, 8.0, 9.5, 1.0, 12.0, 4.0, 5.0, 4.5]\nm = KNeighborsRegressor(n_neighbors=2).fit(X, t)\nprint(m, np.round(m.predict(X[:3]), 6), round(m.score(X, t), 6))\nw = KNeighborsRegressor(3, weights='distance', metric='manhattan').fit(X, t)\nprint(np.round(w.predict([[2.0, 2.0]]), 6), np.round(w.kneighbors([[2.0, 2.0]])[0], 6))\nf = KNeighborsRegressor(2, weights=lambda d: 1 / (d + 1)).fit(X, t)\nprint(np.round(f.predict([[2.0, 2.0]]), 6))",
    "stdout": "KNeighborsRegressor(n_neighbors=2) [2.   2.   8.75] 0.965155\n[1.92887] [[0.7 1.  2.4]]\n[2.065209]\n"
  },
  "knn_k_too_big": {
    "src": "import numpy as np\nfrom sklearn.neighbors import KNeighborsClassifier\nKNeighborsClassifier(n_neighbors=5).fit([[1], [2]], [0, 1]).predict([[1]])",
    "stdout": "",
    "error": "ValueError",
    "message": "Expected n_neighbors <= n_samples_fit, but n_neighbors = 5, n_samples_fit = 2, n_samples = 1"
  },
  "knn_k_zero": {
    "src": "import numpy as np\nfrom sklearn.neighbors import KNeighborsClassifier\nKNeighborsClassifier(n_neighbors=0).fit([[1], [2]], [0, 1])",
    "stdout": "",
    "error": "InvalidParameterError",
    "message": "The 'n_neighbors' parameter of KNeighborsClassifier must be an int in the range [1, inf) or None. Got 0 instead."
  },
  "knn_not_fitted": {
    "src": "import numpy as np\nfrom sklearn.neighbors import KNeighborsClassifier\nKNeighborsClassifier(3).predict([[1]])",
    "stdout": "",
    "error": "NotFittedError",
    "message": "This KNeighborsClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
  },
  "knn_1d": {
    "src": "import numpy as np\nfrom sklearn.neighbors import KNeighborsClassifier\nKNeighborsClassifier(1).fit([1, 2], [0, 1])",
    "stdout": "",
    "error": "ValueError",
    "message": "Expected 2D array, got 1D array instead:\narray=[1 2].\nReshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample."
  },
  "knn_continuous_labels": {
    "src": "import numpy as np\nfrom sklearn.neighbors import KNeighborsClassifier\nKNeighborsClassifier(1).fit([[1], [2]], [0.5, 1.5])",
    "stdout": "",
    "error": "ValueError",
    "message": "Unknown label type: continuous. Maybe you are trying to fit a classifier, which expects discrete classes on a regression target with continuous values."
  },
  "knn_lengths": {
    "src": "import numpy as np\nfrom sklearn.neighbors import KNeighborsClassifier\nKNeighborsClassifier(1).fit([[1], [2]], [0])",
    "stdout": "",
    "error": "ValueError",
    "message": "Found input variables with inconsistent numbers of samples: [2, 1]"
  },
  "knn_features": {
    "src": "import numpy as np\nfrom sklearn.neighbors import KNeighborsClassifier\nKNeighborsClassifier(1).fit([[1, 2], [2, 3]], [0, 1]).predict([[1]])",
    "stdout": "",
    "error": "ValueError",
    "message": "X has 1 features, but KNeighborsClassifier is expecting 2 features as input."
  },
  "knn_nan": {
    "src": "import numpy as np\nfrom sklearn.neighbors import KNeighborsClassifier\nKNeighborsClassifier(1).fit([[1, float('nan')], [2, 3]], [0, 1])",
    "stdout": "",
    "error": "ValueError",
    "message": "Input X contains NaN.\nKNeighborsClassifier does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values"
  },
  "knn_keyword": {
    "src": "import numpy as np\nfrom sklearn.neighbors import KNeighborsClassifier\nKNeighborsClassifier(foo=1)",
    "stdout": "",
    "error": "TypeError",
    "message": "KNeighborsClassifier.__init__() got an unexpected keyword argument 'foo'"
  },
  "knn_method": {
    "src": "import numpy as np\nfrom sklearn.neighbors import KNeighborsClassifier\nprint(KNeighborsClassifier(1).predict_prob)",
    "stdout": "",
    "error": "AttributeError"
  },
  "kmeans_explicit_init_exact": {
    "src": "import numpy as np\nfrom sklearn.cluster import KMeans\nX = np.array([[-1.378, -2.21, -1.414], [0.684, 2.147, -1.254], [-1.996, 0.542, 2.123], [-0.761, 2.892, -1.526], [2.737, 1.312, -3.29], [1.004, -0.448, -0.358], [-1.43, 0.017, -1.281], [0.243, 2.031, 2.631], [2.118, 1.673, 0.826], [-0.404, -2.172, 1.934], [-0.864, 0.342, -2.652], [0.833, 2.647, -0.195], [1.724, -0.304, 0.302], [-0.935, -5.161, 0.685], [0.919, 1.1, 3.267], [1.757, 0.877, -1.281], [5.61, -2.611, -0.745], [-0.958, 2.065, 0.645], [1.04, -3.381, -1.01], [2.004, -0.975, -2.403]])\nm = KMeans(4, init=np.array([[2.71, -1.22, 0.08], [0.24, -0.15, 3.87], [0.56, -2.26, 0.58], [1.27, -2.08, 0.38]]), n_init=1, max_iter=300).fit(X)\nprint(m.cluster_centers_.tolist())\nprint(m.labels_, round(m.inertia_, 9), m.n_iter_)\nprint(m)\nprint(m.predict(X[:3]), m.fit_predict(X)[:3], np.round(m.transform(X[:1]), 9), round(m.score(X), 9))\nprint(KMeans(2, init=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], max_iter=1).fit(X).cluster_centers_.tolist())",
    "stdout": "[[1.228, 1.9246666666666665, -1.12], [-0.44799999999999995, 1.4344999999999999, 2.1665], [-1.0021999999999998, -1.8368000000000002, -0.5456000000000001], [2.2764, -1.5438, -0.8428]]\n[2 0 1 0 0 3 2 1 0 2 2 0 3 2 1 0 3 1 3 3] 90.587498933 4\nKMeans(init=array([[ 2.71, -1.22,  0.08],\n       [ 0.24, -0.15,  3.87],\n       [ 0.56, -2.26,  0.58],\n       [ 1.27, -2.08,  0.38]]),\n       n_clusters=4, n_init=1)\n[2 0 1] [2 0 1] [[4.89623738 5.19300111 1.01716392 3.75828834]] -90.587498933\n[[0.0645, -0.6970833333333332, -0.87275], [1.3966250000000002, 1.0934999999999997, 0.684625]]\n"
  },
  "kmeans_integer_data": {
    "src": "import numpy as np\nfrom sklearn.cluster import KMeans\nX = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])\nm = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(X)\nprint(m.labels_, m.predict([[0, 0], [12, 3]]), m.cluster_centers_, m.inertia_, m.n_iter_)\nprint(KMeans(2, init=np.array([[1, 2], [10, 2]]), n_init=1).fit(X).labels_)",
    "stdout": "[1 1 1 0 0 0] [1 0] [[10.  2.]\n [ 1.  2.]] 16.0 2\n[0 0 0 1 1 1]\n"
  },
  "kmeans_seeded": {
    "src": "import numpy as np\nfrom sklearn.cluster import KMeans\nfrom sklearn.datasets import load_iris\nX = load_iris().data\nm = KMeans(3, random_state=0, n_init=10).fit(X)\nprint(np.round(m.cluster_centers_, 6), m.labels_[:20], round(m.inertia_, 6), m.n_iter_)\nn = KMeans(3, random_state=42).fit(X)\nprint(np.round(n.cluster_centers_, 6), n.labels_[:20], round(n.inertia_, 6), n.n_iter_)\nr = KMeans(n_clusters=4, init='random', random_state=7).fit(X)\nprint(np.round(r.cluster_centers_, 6), r.labels_[:20], round(r.inertia_, 6), r.n_iter_)\nt = KMeans(3, random_state=1, max_iter=2, tol=0.5)\nprint(t.fit_predict(X)[:20], round(t.inertia_, 6), t.n_iter_)",
    "stdout": "[[5.901613 2.748387 4.393548 1.433871]\n [5.006    3.428    1.462    0.246   ]\n [6.85     3.073684 5.742105 2.071053]] [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1] 78.851441 4\n[[6.853846 3.076923 5.715385 2.053846]\n [5.006    3.428    1.462    0.246   ]\n [5.883607 2.740984 4.388525 1.434426]] [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1] 78.855666 11\n[[6.9125   3.1      5.846875 2.13125 ]\n [5.52963  2.622222 3.940741 1.218519]\n [6.236585 2.858537 4.807317 1.621951]\n [5.006    3.428    1.462    0.246   ]] [3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3] 57.256009 7\n[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1] 85.636958 1\n"
  },
  "kmeans_empty_cluster": {
    "src": "import numpy as np\nfrom sklearn.cluster import KMeans\nX = np.array([[0.0], [0.1], [0.2], [10.0], [10.1]])\nm = KMeans(3, init=np.array([[0.1], [10.0], [100.0]]), n_init=1).fit(X)\nprint(m.cluster_centers_.tolist(), m.labels_)",
    "stdout": "[[0.04999999999999982], [10.05], [0.20000000000000018]] [0 0 2 1 1]\n"
  },
  "kmeans_too_few": {
    "src": "import numpy as np\nfrom sklearn.cluster import KMeans\nKMeans(3).fit([[1.0], [2.0]])",
    "stdout": "",
    "error": "ValueError",
    "message": "n_samples=2 should be >= n_clusters=3."
  },
  "kmeans_zero": {
    "src": "import numpy as np\nfrom sklearn.cluster import KMeans\nKMeans(0).fit([[1.0], [2.0]])",
    "stdout": "",
    "error": "InvalidParameterError",
    "message": "The 'n_clusters' parameter of KMeans must be an int in the range [1, inf). Got 0 instead."
  },
  "kmeans_init_shape": {
    "src": "import numpy as np\nfrom sklearn.cluster import KMeans\nKMeans(2, init=np.array([[1.0], [2.0], [3.0]])).fit([[1.0], [2.0]])",
    "stdout": "",
    "error": "ValueError",
    "message": "The shape of the initial centers (3, 1) does not match the number of clusters 2."
  },
  "kmeans_not_fitted": {
    "src": "import numpy as np\nfrom sklearn.cluster import KMeans\nKMeans(2).predict([[1.0], [2.0]])",
    "stdout": "",
    "error": "NotFittedError",
    "message": "This KMeans instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
  },
  "gaussian_nb": {
    "src": "import numpy as np\nX = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [1.0, 0.6], [9.0, 11.0], [8.0, 2.0], [10.0, 2.0], [9.0, 3.0]])\ny = [0, 0, 1, 1, 0, 1, 2, 2, 2]\nfrom sklearn.naive_bayes import GaussianNB\nq = np.array([[1.0, 1.0], [9.0, 9.0], [5.0, 5.0]])\nm = GaussianNB().fit(X, y)\nprint(m, np.round(m.theta_, 8), np.round(m.var_, 8), m.class_count_, np.round(m.class_prior_, 8), m.classes_)\nprint(m.predict(q), np.round(m.predict_proba(q), 6), np.round(m.predict_log_proba(q), 6), round(m.score(X, y), 6))\np = GaussianNB(priors=[0.5, 0.25, 0.25], var_smoothing=0.1).fit(X, y)\nprint(p, np.round(p.class_prior_, 6), np.round(p.var_, 6), np.round(p.predict_proba(q), 6))",
    "stdout": "GaussianNB() [[1.16666667 1.46666667]\n [7.33333333 9.        ]\n [9.         2.33333333]] [[0.05555557 0.38222223]\n [2.8888889  2.00000001]\n [0.66666668 0.22222223]] [3. 3. 3.] [0.33333333 0.33333333 0.33333333] [0 1 2]\n[0 1 1] [[1. 0. 0.]\n [0. 1. 0.]\n [0. 1. 0.]] [[  -0.        -25.210496  -52.436406]\n [-623.204404    0.        -97.687444]\n [-140.835985   -0.        -21.22591 ]] 1.0\nGaussianNB(priors=[0.5, 0.25, 0.25], var_smoothing=0.1) [0.5  0.25 0.25] [[1.311111 1.637778]\n [4.144444 3.255556]\n [1.922222 1.477778]] [[1.00000e+00 0.00000e+00 0.00000e+00]\n [0.00000e+00 9.99999e-01 1.00000e-06]\n [8.53000e-03 9.27533e-01 6.39370e-02]]\n"
  },
  "categorical_nb": {
    "src": "import numpy as np\nfrom sklearn.naive_bayes import CategoricalNB\nX = np.array([[0, 1], [1, 1], [2, 0], [0, 0], [1, 2], [2, 2], [0, 2]])\ny = ['n', 'y', 'y', 'n', 'y', 'n', 'y']\nm = CategoricalNB().fit(X, y)\nprint(m, m.classes_, m.class_count_, np.round(m.class_log_prior_, 8), m.n_categories_)\nprint(m.category_count_, [np.round(a, 8) for a in m.feature_log_prob_])\nq = np.array([[0, 0], [2, 1]])\nprint(m.predict(q), np.round(m.predict_proba(q), 6), round(m.score(X, y), 6))\na = CategoricalNB(alpha=0.5, fit_prior=False, min_categories=4).fit(X, y)\nprint(a, np.round(a.class_log_prior_, 8), a.n_categories_, np.round(a.predict_log_proba(q), 6))\nb = CategoricalNB(class_prior=[0.2, 0.8]).fit(X, y)\nprint(np.round(b.class_log_prior_, 8), b.predict(q))",
    "stdout": "CategoricalNB() ['n' 'y'] [3. 4.] [-0.84729786 -0.55961579] [3 3]\n[array([[2., 0., 1.],\n       [1., 2., 1.]]), array([[1., 1., 1.],\n       [1., 1., 2.]])] [array([[-0.69314718, -1.79175947, -1.09861229],\n       [-1.25276297, -0.84729786, -1.25276297]]), array([[-1.09861229, -1.09861229, -1.09861229],\n       [-1.25276297, -1.25276297, -0.84729786]])]\n['n' 'n'] [[0.604938 0.395062]\n [0.505155 0.494845]] 0.571429\nCategoricalNB(alpha=0.5, fit_prior=False, min_categories=4) [-0.69314718 -0.69314718] [4 4] [[-0.348307 -1.223775]\n [-0.527355 -0.891998]]\n[-1.60943791 -0.22314355] ['y' 'y']\n"
  },
  "categorical_nb_unseen": {
    "src": "import numpy as np\nfrom sklearn.naive_bayes import CategoricalNB\nCategoricalNB().fit([[0], [1]], [0, 1]).predict([[5]])",
    "stdout": "",
    "error": "IndexError",
    "message": "index 5 is out of bounds for axis 1 with size 2"
  },
  "categorical_nb_negative": {
    "src": "import numpy as np\nfrom sklearn.naive_bayes import CategoricalNB\nCategoricalNB().fit([[-1], [1]], [0, 1])",
    "stdout": "",
    "error": "ValueError",
    "message": "Negative values in data passed to CategoricalNB (input X)."
  },
  "multinomial_bernoulli_nb": {
    "src": "import numpy as np\nfrom sklearn.naive_bayes import MultinomialNB, BernoulliNB\nX = np.array([[2, 1, 0], [1, 1, 0], [0, 2, 3], [0, 0, 4], [3, 0, 1]])\ny = [0, 0, 1, 1, 0]\nq = np.array([[1, 0, 2], [4, 0, 0]])\nm = MultinomialNB().fit(X, y)\nprint(m, m.feature_count_, m.class_count_, np.round(m.feature_log_prob_, 8), np.round(m.class_log_prior_, 8))\nprint(m.predict(q), np.round(m.predict_proba(q), 6))\nn = MultinomialNB(alpha=0.5, fit_prior=False).fit(X, y)\nprint(np.round(n.feature_log_prob_, 8), np.round(n.class_log_prior_, 8))\nb = BernoulliNB().fit(X, y)\nprint(b, b.feature_count_, np.round(b.feature_log_prob_, 8), np.round(b.class_log_prior_, 8))\nprint(b.predict(q), np.round(b.predict_proba(q), 6))\nc = BernoulliNB(alpha=2.0, binarize=1.5, class_prior=[0.3, 0.7]).fit(X, y)\nprint(c.feature_count_, np.round(c.feature_log_prob_, 8), np.round(c.class_log_prior_, 8), np.round(c.predict_proba(q), 6))",
    "stdout": "MultinomialNB() [[6. 2. 1.]\n [0. 2. 7.]] [3. 2.] [[-0.5389965  -1.38629436 -1.79175947]\n [-2.48490665 -1.38629436 -0.40546511]] [-0.51082562 -0.91629073]\n[1 0] [[3.96226e-01 6.03774e-01]\n [9.99722e-01 2.78000e-04]]\n[[-0.47957308 -1.43508453 -1.94591015]\n [-3.04452244 -1.43508453 -0.33647224]] [-0.69314718 -0.69314718]\nBernoulliNB() [[3. 2. 1.]\n [0. 1. 2.]] [[-0.22314355 -0.51082562 -0.91629073]\n [-1.38629436 -0.69314718 -0.28768207]] [-0.51082562 -0.91629073]\n[0 0] [[0.671916 0.328084]\n [0.902114 0.097886]]\n[[2. 0. 0.]\n [0. 1. 2.]] [[-0.55961579 -1.25276297 -1.25276297]\n [-1.09861229 -0.69314718 -0.40546511]] [-1.2039728  -0.35667494] [[0.144334 0.855666]\n [0.692219 0.307781]]\n"
  },
  "multinomial_nb_negative": {
    "src": "import numpy as np\nfrom sklearn.naive_bayes import MultinomialNB\nMultinomialNB().fit([[-1.0], [2.0]], [0, 1])",
    "stdout": "",
    "error": "ValueError",
    "message": "Negative values in data passed to MultinomialNB (input X)."
  },
  "gaussian_nb_bad_priors": {
    "src": "import numpy as np\nfrom sklearn.naive_bayes import GaussianNB\nGaussianNB(priors=[0.5, 0.6]).fit([[-1.0], [1.0]], [0, 1])",
    "stdout": "",
    "error": "ValueError",
    "message": "The sum of the priors should be 1."
  },
  "perceptron_fixed_order": {
    "src": "import numpy as np\nfrom sklearn.linear_model import Perceptron\nX = np.array([[2.0, 1.0], [3.0, 4.0], [4.0, 2.0], [3.0, 1.0], [-1.0, -2.0], [-2.0, 0.5], [0.5, -3.0], [-1.5, -1.0]])\ny = [1, 1, 1, 1, 0, 0, 0, 0]\nm = Perceptron(shuffle=False).fit(X, y)\nprint(m, m.coef_.tolist(), m.intercept_.tolist(), m.n_iter_, m.t_, m.classes_)\nprint(m.predict(X[:2]), m.decision_function(X[:2]), m.score(X, y))\nn = Perceptron(shuffle=False, eta0=0.1, max_iter=5, tol=None).fit(X, y)\nprint(n, n.coef_.tolist(), n.intercept_.tolist(), n.n_iter_)\nb = Perceptron(shuffle=False, class_weight={0: 2.0, 1: 0.5}, eta0=0.1).fit(X, y)\nprint(b, b.coef_.tolist(), b.intercept_.tolist(), b.n_iter_)\nc = Perceptron(shuffle=False, class_weight='balanced', fit_intercept=False).fit(X, y)\nprint(c.coef_.tolist(), c.intercept_.tolist())\nl = Perceptron(shuffle=False, penalty='l2', alpha=0.01).fit(X, y)\nprint(np.round(l.coef_, 10), np.round(l.intercept_, 10))",
    "stdout": "Perceptron(shuffle=False) [[2.0, 1.0]] [1.0] 6 49.0 [0 1]\n[1 1] [ 6. 11.] 1.0\nPerceptron(eta0=0.1, max_iter=5, shuffle=False, tol=None) [[0.2, 0.1]] [0.1] 5\nPerceptron(class_weight={0: 2.0, 1: 0.5}, eta0=0.1, shuffle=False) [[0.1, 0.05]] [0.05] 6\n[[2.0, 1.0]] [0.0]\n[[1.24705079 0.62352539]] [1.]\n"
  },
  "perceptron_class_weight_error": {
    "src": "import numpy as np\nfrom sklearn.linear_model import Perceptron\nPerceptron(class_weight={5: 1.0}).fit([[1.0], [2.0]], [0, 1])",
    "stdout": "",
    "error": "ValueError",
    "message": "The classes, [0, 1], are not in class_weight"
  },
  "perceptron_multiclass": {
    "src": "import numpy as np\nX = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [1.0, 0.6], [9.0, 11.0], [8.0, 2.0], [10.0, 2.0], [9.0, 3.0]])\ny = [0, 0, 1, 1, 0, 1, 2, 2, 2]\nfrom sklearn.linear_model import Perceptron\nm = Perceptron(shuffle=False, max_iter=50).fit(X, y)\nprint(m.coef_.tolist(), m.intercept_.tolist(), m.n_iter_, m.t_)\nprint(m.predict(X), m.decision_function(X[:1]).tolist(), m.score(X, y))",
    "stdout": "[[-5.5, 0.599999999999999], [-9.0, 23.6], [4.5, -9.8]] [8.0, -11.0, -2.0] 12 109.0\n[1 1 1 1 0 1 2 2 2] [[3.6999999999999975, 27.200000000000003, -17.1]] 0.7777777777777778\n"
  },
  "perceptron_shuffled_default": {
    "src": "import numpy as np\nX = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [1.0, 0.6], [9.0, 11.0], [8.0, 2.0], [10.0, 2.0], [9.0, 3.0]])\ny = [0, 0, 1, 1, 0, 1, 2, 2, 2]\nfrom sklearn.linear_model import Perceptron\nm = Perceptron().fit(X, y)\nprint(m.coef_.tolist(), m.intercept_.tolist(), m.n_iter_, m.predict(X))\nb = [1, 1, 0, 0, 1, 0, 0, 0, 1]\nn = Perceptron(random_state=5).fit(X, b)\nprint(n.coef_.tolist(), n.intercept_.tolist(), n.n_iter_, n)\nt = Perceptron(random_state=5, class_weight={'a': 1.0, 'b': 3.0}, eta0=0.5).fit(X, ['a' if v else 'b' for v in b])\nprint(t.coef_.tolist(), t.intercept_.tolist(), t.predict(X[:2]), t.classes_)",
    "stdout": "[[-6.5, 1.399999999999999], [-11.0, 16.599999999999998], [5.0, -11.6]] [9.0, -15.0, -1.0] 11 [1 0 1 1 0 1 2 2 2]\n[[1.0, -7.199999999999999]] [18.0] 15 Perceptron(random_state=5)\n[[4.5, 13.699999999999998]] [-12.5] ['b' 'b'] ['a' 'b']\n"
  },
  "linear_regression": {
    "src": "import numpy as np\nfrom sklearn.linear_model import LinearRegression\nX = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 5.0], [4.0, 3.0], [5.0, 8.0]])\ny = [3.1, 4.2, 9.9, 8.7, 14.8]\nm = LinearRegression().fit(X, y)\nprint(m, np.round(m.coef_, 8), round(m.intercept_, 8), m.rank_, np.round(m.singular_, 8))\nprint(np.round(m.predict(X[:2]), 8), round(m.score(X, y), 8))\nn = LinearRegression(fit_intercept=False).fit(X, y)\nprint(np.round(n.coef_, 8), n.intercept_)\nY = np.array([[1.0, 2], [3, 1], [2, 2], [5, 7], [4, 4]])\nw = LinearRegression().fit(X, Y)\nprint(np.round(w.coef_, 8), np.round(w.intercept_, 8), np.round(w.predict(X[:1]), 8), round(w.score(X, Y), 8))\nu = LinearRegression().fit(np.array([[1.0, 2, 3], [2, 1, 0]]), [1.0, 2.0])\nprint(np.round(u.coef_, 6), round(u.intercept_, 6), u.rank_)\nd = LinearRegression().fit(np.array([[1.0, 2], [2, 4], [3, 6]]), [1.0, 2, 3])\nprint(np.round(d.coef_, 6), round(d.intercept_ + 0.0, 6), d.rank_)",
    "stdout": "LinearRegression() [1.455      0.95357143] 0.15142857 2 [6.15143751 1.72041173]\n[3.51357143 4.015     ] 0.9924812\n[1.5025641  0.94871795] 0.0\n[[ 1.575      -0.55357143]\n [ 1.85       -0.60714286]] [ 0.37857143 -0.04285714] [[0.84642857 0.59285714]] 0.80144424\n[ 0.090909 -0.090909 -0.272727] 1.909091 1\n[0.2 0.4] 0.0 1\n"
  },
  "ridge": {
    "src": "import numpy as np\nfrom sklearn.linear_model import Ridge\nX = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 5.0], [4.0, 3.0], [5.0, 8.0]])\ny = [3.1, 4.2, 9.9, 8.7, 14.8]\nm = Ridge().fit(X, y)\nprint(m, np.round(m.coef_, 8), round(m.intercept_, 8), np.round(m.predict(X[:2]), 8), round(m.score(X, y), 8))\nn = Ridge(alpha=10.0, fit_intercept=False).fit(X, y)\nprint(n, np.round(n.coef_, 8), n.intercept_)\nu = Ridge(0.5).fit(np.array([[1.0, 2, 3], [2, 1, 0]]), [1.0, 2.0])\nprint(np.round(u.coef_, 6), round(u.intercept_, 6))",
    "stdout": "Ridge() [1.24096229 1.01781534] 0.54941482 [3.8260078  4.04915475] 0.99022368\nRidge(alpha=10.0, fit_intercept=False) [1.05798611 1.14409722] 0.0\n[ 0.083333 -0.083333 -0.25    ] 1.875\n"
  },
  "logistic_binary_within_tolerance": {
    "src": "import numpy as np\nfrom sklearn.linear_model import LogisticRegression\nX = np.array([[0.5, 1.0], [1.0, 2.5], [2.0, 1.0], [3.0, 3.5], [4.0, 3.0], [5.0, 4.5], [1.5, 3.0], [3.5, 0.5]])\ny = [0, 0, 0, 1, 1, 1, 1, 0]\nm = LogisticRegression(max_iter=10000, tol=1e-8).fit(X, y)\nprint(m, np.round(m.coef_, 3), np.round(m.intercept_, 3), m.classes_)\nprint(m.predict(X), np.round(m.predict_proba(X[:2]), 3), np.round(m.decision_function(X[:2]), 3), m.score(X, y))\nc = LogisticRegression(C=0.1, class_weight='balanced', max_iter=10000, tol=1e-8).fit(X, y)\nprint(c, np.round(c.coef_, 3), np.round(c.intercept_, 3))\nf = LogisticRegression(fit_intercept=False, max_iter=10000, tol=1e-8).fit(X, ['p' if v else 'q' for v in y])\nprint(np.round(f.coef_, 3), f.intercept_, f.classes_, f.predict(X[:2]))",
    "stdout": "LogisticRegression(max_iter=10000, tol=1e-08) [[0.535 1.167]] [-3.998] [0 1]\n[0 0 0 1 1 1 1 0] [[0.929 0.071]\n [0.633 0.367]] [-2.564 -0.546] 1.0\nLogisticRegression(C=0.1, class_weight='balanced', max_iter=10000, tol=1e-08) [[0.192 0.313]] [-1.229]\n[[ 0.131 -0.428]] [0.] ['p' 'q'] ['p' 'p']\n"
  },
  "logistic_multinomial_within_tolerance": {
    "src": "import numpy as np\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.datasets import load_iris\nX, y = load_iris(return_X_y=True)\nm = LogisticRegression(max_iter=10000, tol=1e-9).fit(X, y)\nprint(np.round(m.coef_, 3), np.round(m.intercept_, 3), round(m.score(X, y), 4))\nprint(np.round(m.predict_proba(X[:2]), 3), m.predict(X[:2]))",
    "stdout": "[[-0.424  0.967 -2.517 -1.079]\n [ 0.534 -0.322 -0.206 -0.944]\n [-0.111 -0.646  2.724  2.024]] [  9.85    2.237 -12.087] 0.9733\n[[0.982 0.018 0.   ]\n [0.971 0.029 0.   ]] [0 0]\n"
  },
  "decision_tree_exact": {
    "src": "import numpy as np\nfrom sklearn.tree import DecisionTreeClassifier, export_text\nfrom sklearn.datasets import load_iris\nX, y = load_iris(return_X_y=True)\nfor s in [0, 1, 42]:\n    m = DecisionTreeClassifier(random_state=s).fit(X, y)\n    print(m.tree_.feature, np.round(m.tree_.threshold, 6), m.get_depth(), m.get_n_leaves(), np.round(m.feature_importances_, 6))\nn = DecisionTreeClassifier(criterion='entropy', max_depth=2, min_samples_leaf=5, random_state=0).fit(X, y)\nprint(n, n.predict(X[:3]), np.round(n.predict_proba(X[48:52]), 6), round(n.score(X, y), 6), n.n_classes_, n.classes_)\nprint(export_text(n))\nprint(export_text(n, feature_names=['a', 'b', 'c', 'd'], decimals=3))\nt = DecisionTreeClassifier(min_samples_split=20, random_state=3).fit(X[:, :2], ['s' if v else 't' for v in y])\nprint(t.get_depth(), t.predict(X[:2, :2]), t.apply(X[:2, :2]), t.tree_.node_count, t.tree_.children_left[:3], t.tree_.n_node_samples[:3], np.round(t.tree_.impurity[:3], 6))",
    "stdout": "[ 3 -2  3  2  3 -2 -2  3 -2  2 -2 -2  2  1 -2 -2 -2] [ 0.8  -2.    1.75  4.95  1.65 -2.   -2.    1.55 -2.    5.45 -2.   -2.\n  4.85  3.1  -2.   -2.   -2.  ] 5 9 [0.       0.013333 0.064056 0.922611]\n[ 3 -2  3  2  3 -2 -2  3 -2  0 -2 -2  2  0 -2 -2 -2] [ 0.8  -2.    1.75  4.95  1.65 -2.   -2.    1.55 -2.    6.95 -2.   -2.\n  4.85  5.95 -2.   -2.   -2.  ] 5 9 [0.026667 0.       0.050723 0.922611]\n[ 2 -2  3  2  3 -2 -2  3 -2  2 -2 -2  2  0 -2 -2 -2] [ 2.45 -2.    1.75  4.95  1.65 -2.   -2.    1.55 -2.    5.45 -2.   -2.\n  4.85  5.95 -2.   -2.   -2.  ] 5 9 [0.013333 0.       0.564056 0.422611]\nDecisionTreeClassifier(criterion='entropy', max_depth=2, min_samples_leaf=5,\n                       random_state=0) [0 0 0] [[1.       0.       0.      ]\n [1.       0.       0.      ]\n [0.       0.907407 0.092593]\n [0.       0.907407 0.092593]] 0.96 3 [0 1 2]\n|--- feature_3 <= 0.80\n|   |--- class: 0\n|--- feature_3 >  0.80\n|   |--- feature_3 <= 1.75\n|   |   |--- class: 1\n|   |--- feature_3 >  1.75\n|   |   |--- class: 2\n\n|--- d <= 0.800\n|   |--- class: 0\n|--- d >  0.800\n|   |--- d <= 1.750\n|   |   |--- class: 1\n|   |--- d >  1.750\n|   |   |--- class: 2\n\n3 ['t' 't'] [4 4] 9 [ 1  2 -1] [150  52   7] [0.444444 0.232988 0.244898]\n"
  },
  "pca_sign_as_sklearn": {
    "src": "import numpy as np\nfrom sklearn.decomposition import PCA\nfrom sklearn.datasets import load_iris\nX = load_iris().data\nm = PCA(2).fit(X)\nprint(m, np.round(m.components_, 6), np.round(m.explained_variance_, 6), np.round(m.explained_variance_ratio_, 6), np.round(m.singular_values_, 6), np.round(m.mean_, 6), m.n_components_, round(m.noise_variance_, 6))\nprint(np.round(m.transform(X[:3]), 6), np.round(m.inverse_transform(m.transform(X[:1])), 6))\nw = PCA(n_components=0.95, whiten=True)\nZ = w.fit_transform(X)\nprint(w, w.n_components_, np.round(Z[:2], 6), np.round(w.inverse_transform(Z[:1]), 6))\nS = np.array([[1.0, 2, 0], [2, 1, 3], [0, 1, 5], [4, 0, 1]])\nf = PCA()\nprint(np.round(f.fit_transform(S), 5), np.round(f.components_, 5), f.n_components_, np.round(f.get_covariance(), 5))\nprint(np.round(PCA(1).fit(np.array([[1.0, 2, 3, 4], [4, 3, 1, 0]])).components_, 5))",
    "stdout": "PCA(n_components=2) [[ 0.361387 -0.084523  0.856671  0.358289]\n [ 0.656589  0.730161 -0.173373 -0.075481]] [4.228242 0.242671] [0.924619 0.053066] [25.09996   6.013147] [5.843333 3.057333 3.758    1.199333] 2 0.051022\n[[-2.684126  0.319397]\n [-2.714142 -0.177001]\n [-2.888991 -0.144949]] [[5.083039 3.517414 1.403214 0.213532]]\nPCA(n_components=0.95, whiten=True) 2 [[-1.305338  0.648369]\n [-1.319935 -0.359309]] [[5.083039 3.517414 1.403214 0.213532]]\n[[-1.46936 -2.11318 -0.0214 ]\n [ 0.50479  0.53333  0.29283]\n [ 3.2561   0.04435 -0.14443]\n [-2.29153  1.5355  -0.127  ]] [[-0.52695  0.04501  0.8487 ]\n [ 0.71669 -0.5132   0.47221]\n [ 0.4568   0.85709  0.23817]] 3 [[ 2.91667 -1.      -1.91667]\n [-1.       0.66667 -0.33333]\n [-1.91667 -0.33333  4.91667]]\n[[-0.54772 -0.18257  0.36515  0.7303 ]]\n"
  },
  "pca_too_many": {
    "src": "import numpy as np\nfrom sklearn.decomposition import PCA\nPCA(n_components=5).fit(np.zeros((3, 2)))",
    "stdout": "",
    "error": "ValueError",
    "message": "n_components=5 must be between 0 and min(n_samples, n_features)=2 with svd_solver='full'"
  },
  "metrics_classification": {
    "src": "import numpy as np\nfrom sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, fbeta_score, precision_recall_fscore_support\nt = [0, 1, 2, 2, 1, 0, 2, 1, 1, 0]\np = [0, 2, 2, 2, 1, 1, 0, 1, 1, 0]\nprint(accuracy_score(t, p), accuracy_score(t, p, normalize=False))\nprint(confusion_matrix(t, p))\nprint(confusion_matrix(t, p, labels=[2, 0]))\nprint(confusion_matrix(t, p, normalize='true'))\nprint(confusion_matrix(t, p, normalize='pred'), confusion_matrix(t, p, normalize='all'))\nfor a in ['macro', 'micro', 'weighted']:\n    print(a, precision_score(t, p, average=a), recall_score(t, p, average=a), f1_score(t, p, average=a), fbeta_score(t, p, beta=2, average=a))\nprint(precision_score(t, p, average=None), recall_score(t, p, average=None), f1_score(t, p, average=None))\nprint(precision_recall_fscore_support(t, p))\nb = [1, 0, 1, 1, 0, 0, 1, 0]\nq = [1, 1, 1, 0, 0, 0, 1, 1]\nprint(precision_score(b, q), recall_score(b, q), f1_score(b, q), f1_score(b, q, pos_label=0), fbeta_score(b, q, beta=0.5), precision_recall_fscore_support(b, q, average='binary'))\nprint(precision_score([0, 0], [0, 0]), precision_score([1, 0], [0, 0], zero_division=1), recall_score([0, 0], [1, 1], zero_division=1))\nw = ['cat', 'dog', 'cat', 'bird']\nv = ['cat', 'cat', 'cat', 'dog']\nprint(accuracy_score(w, v), confusion_matrix(w, v), f1_score(w, v, average='macro'), f1_score(w, v, labels=['cat'], average='macro'))\nprint(precision_score(['y', 'n', 'y'], ['y', 'y', 'n'], pos_label='y'))",
    "stdout": "0.7 7.0\n[[2 1 0]\n [0 3 1]\n [1 0 2]]\n[[2 1]\n [0 2]]\n[[0.66666667 0.33333333 0.        ]\n [0.         0.75       0.25      ]\n [0.33333333 0.         0.66666667]]\n[[0.66666667 0.25       0.        ]\n [0.         0.75       0.33333333]\n [0.33333333 0.         0.66666667]] [[0.2 0.1 0. ]\n [0.  0.3 0.1]\n [0.1 0.  0.2]]\nmacro 0.6944444444444443 0.6944444444444443 0.6944444444444443 0.6944444444444443\nmicro 0.7 0.7 0.7 0.7\nweighted 0.7 0.7 0.7 0.7\n[0.66666667 0.75       0.66666667] [0.66666667 0.75       0.66666667] [0.66666667 0.75       0.66666667]\n(array([0.66666667, 0.75      , 0.66666667]), array([0.66666667, 0.75      , 0.66666667]), array([0.66666667, 0.75      , 0.66666667]), array([3, 4, 3]))\n0.6 0.75 0.6666666666666666 0.5714285714285714 0.625 (0.6, 0.75, 0.6666666666666666, None)\n0.0 1.0 1.0\n0.5 [[0 0 1]\n [0 2 0]\n [0 1 0]] 0.26666666666666666 0.8\n0.5\n"
  },
  "classification_report_exact": {
    "src": "import numpy as np\nfrom sklearn.metrics import classification_report\nprint(classification_report([0, 1, 2, 2, 1, 0, 2, 1, 1, 0], [0, 2, 2, 2, 1, 1, 0, 1, 1, 0]))\nprint(classification_report([0, 1, 1, 0, 1, 1, 0, 1], [0, 1, 1, 1, 1, 1, 0, 0], digits=3))\nprint(classification_report([1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0, 0, 0], digits=1))\nprint(classification_report(['cat', 'dog', 'cat'], ['cat', 'cat', 'cat'], zero_division=0))\nprint(classification_report([0, 1, 1], [1, 0, 0], target_names=['negative', 'positive']))\nprint(classification_report([0, 1, 2], [0, 1, 1], labels=[0, 1]))\nprint(classification_report([0, 1, 2], [0, 1, 1], labels=[0, 1, 2], output_dict=True))\nprint(classification_report([1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0]))",
    "stdout": "              precision    recall  f1-score   support\n\n           0       0.67      0.67      0.67         3\n           1       0.75      0.75      0.75         4\n           2       0.67      0.67      0.67         3\n\n    accuracy                           0.70        10\n   macro avg       0.69      0.69      0.69        10\nweighted avg       0.70      0.70      0.70        10\n\n              precision    recall  f1-score   support\n\n           0      0.667     0.667     0.667         3\n           1      0.800     0.800     0.800         5\n\n    accuracy                          0.750         8\n   macro avg      0.733     0.733     0.733         8\nweighted avg      0.750     0.750     0.750         8\n\n              precision    recall  f1-score   support\n\n           0        0.4       1.0       0.6         2\n           1        1.0       0.6       0.8         8\n\n    accuracy                            0.7        10\n   macro avg        0.7       0.8       0.7        10\nweighted avg        0.9       0.7       0.7        10\n\n              precision    recall  f1-score   support\n\n         cat       0.67      1.00      0.80         2\n         dog       0.00      0.00      0.00         1\n\n    accuracy                           0.67         3\n   macro avg       0.33      0.50      0.40         3\nweighted avg       0.44      0.67      0.53         3\n\n              precision    recall  f1-score   support\n\n    negative       0.00      0.00      0.00       1.0\n    positive       0.00      0.00      0.00       2.0\n\n    accuracy                           0.00       3.0\n   macro avg       0.00      0.00      0.00       3.0\nweighted avg       0.00      0.00      0.00       3.0\n\n              precision    recall  f1-score   support\n\n           0       1.00      1.00      1.00         1\n           1       0.50      1.00      0.67         1\n\n   micro avg       0.67      1.00      0.80         2\n   macro avg       0.75      1.00      0.83         2\nweighted avg       0.75      1.00      0.83         2\n\n{'0': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 1.0}, '1': {'precision': 0.5, 'recall': 1.0, 'f1-score': 0.6666666666666666, 'support': 1.0}, '2': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 1.0}, 'accuracy': 0.6666666666666666, 'macro avg': {'precision': 0.5, 'recall': 0.6666666666666666, 'f1-score': 0.5555555555555555, 'support': 3.0}, 'weighted avg': {'precision': 0.5, 'recall': 0.6666666666666666, 'f1-score': 0.5555555555555555, 'support': 3.0}}\n              precision    recall  f1-score   support\n\n           0       1.00      0.60      0.75         5\n           1       0.60      1.00      0.75         3\n\n    accuracy                           0.75         8\n   macro avg       0.80      0.80      0.75         8\nweighted avg       0.85      0.75      0.75         8\n\n"
  },
  "regression_metrics": {
    "src": "import numpy as np\nfrom sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error\nt = [6.025, 1.455, 2.147, -3.372, -0.048, 3.806, 1.326, 0.721, 4.851, -1.993, 1.391, 0.629, 3.55, 2.554, 1.888]\np = [-5.256, -0.154, 1.328, 0.965, -9.167, -3.486, 1.523, 2.559, 1.362, -2.8, 2.078, 2.761, 0.665, -1.08, 0.37]\nprint(mean_squared_error(t, p), mean_absolute_error(t, p), r2_score(t, p), root_mean_squared_error(t, p))\nprint(r2_score([1.0, 1.0], [1.0, 1.0]), r2_score([1.0, 1.0], [1.0, 2.0]), mean_squared_error([[1.0, 2.0], [3.0, 4.0]], [[1.5, 2.0], [3.0, 3.0]]), r2_score(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]]), np.array([[1.5, 2.0], [3.0, 3.0], [5.0, 7.0]])))",
    "stdout": "22.050127866666667 3.4429333333333334 -3.0210666196971525 4.695756367899283\n1.0 0.0 0.3125 0.9449013157894737\n"
  },
  "roc": {
    "src": "import numpy as np\nfrom sklearn.metrics import roc_auc_score, roc_curve, auc\nt = [1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0]\ns = [0.7, 0.07, 0.14, 0.79, 0.14, 0.43, 0.57, 0.45, 1.0, 0.65, 1.0, 0.12, 0.42, 0.67, 0.91, 0.67, 0.96, 0.45, 0.96, 0.27, 0.07]\nprint(round(roc_auc_score(t, s), 10))\nf, r, th = roc_curve(t, s)\nprint(np.round(f, 8), np.round(r, 8), np.round(th, 8), round(auc(f, r), 10))\nprint(roc_auc_score(['a', 'b', 'a', 'b'], [0.1, 0.4, 0.35, 0.8]), roc_auc_score([1, 1], [0.1, 0.2]), roc_curve([0, 1], [0.2, 0.9])[2], auc([0, 1, 2], [0, 1, 0]))",
    "stdout": "0.7090909091\n[0.         0.09090909 0.09090909 0.09090909 0.18181818 0.36363636\n 0.45454545 0.72727273 0.90909091 0.90909091 1.        ] [0.  0.1 0.3 0.6 0.7 0.7 0.8 0.8 0.8 0.9 1. ] [ inf 1.   0.96 0.7  0.67 0.57 0.45 0.27 0.14 0.12 0.07] 0.7090909091\n1.0 nan [inf 0.9 0.2] 1.0\n"
  },
  "silhouette": {
    "src": "import numpy as np\nfrom sklearn.metrics import silhouette_score\nX = np.array([[1.0, 2], [1, 3], [5, 5], [6, 5], [6, 7], [0, 0]])\nprint(round(silhouette_score(X, [0, 0, 1, 1, 1, 0]), 10), round(silhouette_score(X, ['a', 'b', 'b', 'c', 'c', 'a'], metric='euclidean'), 10), round(silhouette_score(X, [0, 1, 1, 1, 1, 1]), 10))",
    "stdout": "0.6998293256 0.0925549675 -0.1002386818\n"
  },
  "metrics_lengths": {
    "src": "import numpy as np\nfrom sklearn.metrics import accuracy_score\naccuracy_score([0, 1], [0])",
    "stdout": "",
    "error": "ValueError",
    "message": "Found input variables with inconsistent numbers of samples: [2, 1]"
  },
  "metrics_multiclass_binary_average": {
    "src": "import numpy as np\nfrom sklearn.metrics import f1_score\nf1_score([0, 1, 2], [0, 1, 1])",
    "stdout": "",
    "error": "ValueError",
    "message": "Target is multiclass but average='binary'. Please choose another average setting, one of [None, 'micro', 'macro', 'weighted']."
  },
  "metrics_continuous": {
    "src": "import numpy as np\nfrom sklearn.metrics import accuracy_score\naccuracy_score([0.5, 1.5], [0.5, 1.0])",
    "stdout": "",
    "error": "ValueError",
    "message": "continuous is not supported"
  },
  "metrics_mix": {
    "src": "import numpy as np\nfrom sklearn.metrics import accuracy_score\naccuracy_score([0, 1], [0.5, 1.0])",
    "stdout": "",
    "error": "ValueError",
    "message": "Classification metrics can't handle a mix of binary and continuous targets"
  },
  "metrics_roc_multiclass": {
    "src": "import numpy as np\nfrom sklearn.metrics import roc_auc_score\nroc_auc_score([0, 1, 2], [0.1, 0.2, 0.3])",
    "stdout": "",
    "error": "ValueError",
    "message": "multi_class must be in ('ovo', 'ovr')"
  },
  "metrics_mse_lengths": {
    "src": "import numpy as np\nfrom sklearn.metrics import mean_squared_error\nmean_squared_error([1.0], [1.0, 2.0])",
    "stdout": "",
    "error": "ValueError",
    "message": "Found input variables with inconsistent numbers of samples: [1, 2]"
  },
  "metrics_mse_squared": {
    "src": "import numpy as np\nfrom sklearn.metrics import mean_squared_error\nmean_squared_error([1.0, 2.0], [1.0, 2.0], squared=False)",
    "stdout": "",
    "error": "TypeError",
    "message": "got an unexpected keyword argument 'squared'"
  },
  "metrics_binary_target": {
    "src": "import numpy as np\nfrom sklearn.metrics import precision_score\nprecision_score([1, 2], [1, 2])",
    "stdout": ""
  },
  "metrics_bad_pos_label": {
    "src": "import numpy as np\nfrom sklearn.metrics import precision_score\nprecision_score([0, 1], [0, 1], pos_label=5)",
    "stdout": "",
    "error": "ValueError",
    "message": "pos_label=5 is not a valid label. It should be one of [0, 1]"
  },
  "iris": {
    "src": "import numpy as np\nfrom sklearn.datasets import load_iris\nd = load_iris()\nprint(d.data.shape, d.target.shape, d.target_names, d.feature_names, d.data[:3], d.target[48:52], list(d.keys())[:2], d['target_names'][0], d.filename)\nprint(np.round(d.data.mean(axis=0), 6), repr(d.target_names))\nX, y = load_iris(return_X_y=True)\nprint(X.shape, y.shape, X[149], y[149], X[50], X[100])\nprint(np.round(X.std(axis=0), 6), np.bincount(y))",
    "stdout": "(150, 4) (150,) ['setosa' 'versicolor' 'virginica'] ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'] [[5.1 3.5 1.4 0.2]\n [4.9 3.  1.4 0.2]\n [4.7 3.2 1.3 0.2]] [0 0 1 1] ['data', 'target'] setosa iris.csv\n[5.843333 3.057333 3.758    1.199333] array(['setosa', 'versicolor', 'virginica'], dtype='<U10')\n(150, 4) (150,) [5.9 3.  5.1 1.8] 2 [7.  3.2 4.7 1.4] [6.3 3.3 6.  2.5]\n[0.825301 0.434411 1.759404 0.759693] [50 50 50]\n"
  },
  "make_blobs_exact": {
    "src": "import numpy as np\nfrom sklearn.datasets import make_blobs\nX, y = make_blobs(n_samples=12, centers=3, n_features=2, random_state=0)\nprint(X.shape, y, np.round(X, 8))\nA, b, c = make_blobs(n_samples=[3, 4], centers=[[0.0, 0.0], [5.0, 5.0]], cluster_std=[0.5, 1.0], random_state=1, return_centers=True, shuffle=False)\nprint(np.round(A, 8), b, c)\nprint(make_blobs(random_state=2)[0].shape, make_blobs(n_samples=5, n_features=4, centers=2, cluster_std=2.5, center_box=(0, 1), random_state=3)[1])",
    "stdout": "(12, 2) [1 0 2 0 0 1 1 2 0 2 2 1] [[ 3.54934659  0.6925054 ]\n [ 1.9263585   4.15243012]\n [ 0.0058752   4.38724103]\n [ 1.12031365  5.75806083]\n [ 1.7373078   4.42546234]\n [ 2.36833522  0.04356792]\n [-0.49772229  1.55128226]\n [-1.4811455   2.73069841]\n [ 0.87305123  4.71438583]\n [-0.66246781  2.17571724]\n [ 0.74285061  1.46351659]\n [ 2.49913075  1.23133799]]\n[[ 0.81217268 -0.30587821]\n [-0.26408588 -0.53648431]\n [ 0.43270381 -1.15076935]\n [ 6.74481176  4.2387931 ]\n [ 5.3190391   4.75062962]\n [ 6.46210794  2.93985929]\n [ 4.6775828   4.61594565]] [0 0 0 1 1 1 1] [[0. 0.]\n [5. 5.]]\n(100, 2) [0 1 0 0 1]\n"
  },
  "make_classification": {
    "src": "import numpy as np\nfrom sklearn.datasets import make_classification\nX, y = make_classification(n_samples=10, random_state=0)\nprint(X.shape, y, np.round(X[:2], 3))\nA, b = make_classification(n_samples=30, n_features=5, n_informative=3, n_redundant=1, n_repeated=1, n_classes=3, n_clusters_per_class=1, random_state=1)\nprint(A.shape, b, np.round(A[:2], 3))\nC, d = make_classification(n_samples=20, n_features=4, n_informative=2, n_redundant=0, flip_y=0, class_sep=2.0, weights=[0.8], shuffle=False, random_state=2)\nprint(C.shape, d, np.round(C[:2], 3))\nE, f = make_classification(n_samples=8, n_features=3, n_informative=2, n_redundant=0, shift=None, scale=None, random_state=3)\nprint(np.round(E[:2], 3), f)",
    "stdout": "(10, 20) [0 1 0 1 1 0 0 0 1 1] [[ 1.896 -0.871 -1.165  0.828  0.129  1.488  0.901 -0.579  0.466  0.056\n   0.402 -2.375 -0.312 -0.663 -0.685  0.729 -1.235 -1.536  1.139  2.371]\n [-0.155  1.923 -0.861  1.     0.969  0.947  1.91   1.481 -0.268  0.906\n  -0.414 -0.987  1.868 -0.057 -0.747 -1.27   1.944  0.802 -1.173  1.21 ]]\n(30, 5) [1 0 0 0 0 2 1 2 1 2 1 2 1 0 1 2 2 2 1 1 0 0 2 0 0 1 1 2 2 0] [[ 0.377 -0.601  0.883  0.377 -0.486]\n [-0.541  0.581  0.539 -0.541  0.438]]\n(20, 4) [0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 1] [[-2.332 -1.86   0.433  1.278]\n [-3.444 -1.561 -0.635  0.508]]\n[[-115.176   -7.516   -8.059]\n [  -4.667  -22.269    7.015]] [0 0 1 1 1 0 0 1]\n"
  },
  "estimator_reprs": {
    "src": "import numpy as np\nfrom sklearn.linear_model import LogisticRegression, Perceptron, LinearRegression, Ridge\nfrom sklearn.cluster import KMeans\nfrom sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor\nfrom sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder, PolynomialFeatures, Normalizer, OrdinalEncoder\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.naive_bayes import CategoricalNB, GaussianNB, MultinomialNB, BernoulliNB\nfrom sklearn.decomposition import PCA\nprint(LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=0, solver='newton-cg', tol=0.01))\nprint(KMeans(n_clusters=3, init='random', n_init=10, max_iter=100, tol=0.001, random_state=42, verbose=0))\nprint(KNeighborsClassifier(n_neighbors=3, weights='distance', metric='manhattan', p=1, algorithm='brute', leaf_size=10))\nprint(OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))\nprint(DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=2, min_samples_split=4, random_state=7, ccp_alpha=0.1))\nprint(Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier(n_neighbors=3))]))\nprint(Pipeline(steps=[('scaler', StandardScaler()), ('minmax', MinMaxScaler(feature_range=(0, 2))), ('clf', LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=0))]))\nprint(ColumnTransformer([('a', StandardScaler(), [0, 1]), ('b', OneHotEncoder(handle_unknown='ignore'), ['x', 'y', 'z'])], remainder='passthrough'))\nprint(ColumnTransformer([('num', Pipeline([('s', StandardScaler()), ('m', MinMaxScaler())]), ['age', 'pay', 'height', 'weight', 'income', 'score']), ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['city'])]))\nprint(GaussianNB(), CategoricalNB(alpha=0.5), Perceptron(class_weight={0: 1.0, 1: 2.0}, eta0=0.1), GaussianNB(priors=[0.5, 0.5]), MultinomialNB(fit_prior=False), BernoulliNB(binarize=None))\nprint(repr(KMeans()), str(KMeans(2)), LabelEncoder(), PolynomialFeatures(3), Normalizer('l1'), OrdinalEncoder(), PCA(2), LinearRegression(), Ridge(0.5), KNeighborsRegressor(2))\nprint(KNeighborsClassifier(3).get_params())\nprint(StandardScaler().get_params(deep=False), KNeighborsClassifier(3).set_params(n_neighbors=4))\nk = KNeighborsClassifier(3)\nprint(k.n_neighbors)\nk.n_neighbors = 4\nprint(k, k.weights)",
    "stdout": "LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000,\n                   random_state=0, solver='newton-cg', tol=0.01)\nKMeans(init='random', max_iter=100, n_clusters=3, n_init=10, random_state=42,\n       tol=0.001)\nKNeighborsClassifier(algorithm='brute', leaf_size=10, metric='manhattan',\n                     n_neighbors=3, p=1, weights='distance')\nOneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)\nDecisionTreeClassifier(ccp_alpha=0.1, criterion='entropy', max_depth=3,\n                       min_samples_leaf=2, min_samples_split=4, random_state=7)\nPipeline(steps=[('scaler', StandardScaler()),\n                ('clf', KNeighborsClassifier(n_neighbors=3))])\nPipeline(steps=[('scaler', StandardScaler()),\n                ('minmax', MinMaxScaler(feature_range=(0, 2))),\n                ('clf',\n                 LogisticRegression(C=0.1, class_weight='balanced',\n                                    max_iter=1000, random_state=0))])\nColumnTransformer(remainder='passthrough',\n                  transformers=[('a', StandardScaler(), [0, 1]),\n                                ('b', OneHotEncoder(handle_unknown='ignore'),\n                                 ['x', 'y', 'z'])])\nColumnTransformer(transformers=[('num',\n                                 Pipeline(steps=[('s', StandardScaler()),\n                                                 ('m', MinMaxScaler())]),\n                                 ['age', 'pay', 'height', 'weight', 'income',\n                                  'score']),\n                                ('cat',\n                                 OneHotEncoder(handle_unknown='ignore',\n                                               sparse_output=False),\n                                 ['city'])])\nGaussianNB() CategoricalNB(alpha=0.5) Perceptron(class_weight={0: 1.0, 1: 2.0}, eta0=0.1) GaussianNB(priors=[0.5, 0.5]) MultinomialNB(fit_prior=False) BernoulliNB(binarize=None)\nKMeans() KMeans(n_clusters=2) LabelEncoder() PolynomialFeatures(degree=3) Normalizer(norm='l1') OrdinalEncoder() PCA(n_components=2) LinearRegression() Ridge(alpha=0.5) KNeighborsRegressor(n_neighbors=2)\n{'algorithm': 'auto', 'leaf_size': 30, 'metric': 'minkowski', 'metric_params': None, 'n_jobs': None, 'n_neighbors': 3, 'p': 2, 'weights': 'uniform'}\n{'copy': True, 'with_mean': True, 'with_std': True} KNeighborsClassifier(n_neighbors=4)\n3\nKNeighborsClassifier(n_neighbors=4) uniform\n"
  },
  "clone": {
    "src": "import numpy as np\nfrom sklearn.base import clone\nfrom sklearn.neighbors import KNeighborsClassifier\nk = KNeighborsClassifier(3).fit([[1.0], [2.0], [3.0]], [0, 1, 1])\nc = clone(k)\nprint(c, hasattr(c, 'classes_'), hasattr(k, 'classes_'))",
    "stdout": "KNeighborsClassifier(n_neighbors=3) False True\n"
  },
  "set_params_error": {
    "src": "import numpy as np\nfrom sklearn.neighbors import KNeighborsClassifier\nKNeighborsClassifier().set_params(foo=1)",
    "stdout": "",
    "error": "ValueError",
    "message": "Invalid parameter 'foo' for estimator KNeighborsClassifier(). Valid parameters are: ['algorithm', 'leaf_size', 'metric', 'metric_params', 'n_jobs', 'n_neighbors', 'p', 'weights']."
  },
  "except_catches_sklearn_errors": {
    "src": "import numpy as np\nfrom sklearn.exceptions import NotFittedError\nfrom sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.cluster import KMeans\ntry:\n    KNeighborsClassifier().predict([[1]])\nexcept NotFittedError as e:\n    print(type(e).__name__, e)\ntry:\n    KNeighborsClassifier().predict([[1]])\nexcept ValueError as e:\n    print('ValueError too')\ntry:\n    KMeans(0).fit([[1.0]])\nexcept ValueError as e:\n    print(type(e).__name__, e)",
    "stdout": "NotFittedError This KNeighborsClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.\nValueError too\nInvalidParameterError The 'n_clusters' parameter of KMeans must be an int in the range [1, inf). Got 0 instead.\n"
  }
}

describe('scikit-learn matches', () => {
  for (const [name, c] of Object.entries(CASES)) {
    it(name, () => {
      const r = run(c.src)
      expect(r.error?.type ?? null).toBe(c.error ?? null)
      if (c.message) expect(r.error?.message).toBe(c.message)
      expect(r.stdout).toBe(c.stdout)
    })
  }
})

const H = 'import numpy as np\n'

describe('options that would change the answer and are not implemented say so', () => {
  const cases: [string, string, RegExp][] = [
    ['from sklearn.cluster import KMeans', "KMeans(2, algorithm='elkan').fit([[1.0], [2.0]])", /algorithm='elkan'/],
    ['from sklearn.cluster import KMeans', 'KMeans(1).fit([[1.0], [2.0]], sample_weight=[1, 2])', /sample_weight/],
    ['from sklearn.neighbors import KNeighborsClassifier', "KNeighborsClassifier(1, metric=lambda a, b: 0.0).fit([[1.0], [2.0]], [0, 1])", /named metrics/],
    ['from sklearn.neighbors import KNeighborsClassifier', "KNeighborsClassifier(1, metric='cosine').fit([[1.0], [2.0]], [0, 1])", /metric='cosine'/],
    ['from sklearn.linear_model import Perceptron', "Perceptron(penalty='l1').fit([[1.0], [2.0]], [0, 1])", /penalty='l1'/],
    ['from sklearn.linear_model import Perceptron', 'Perceptron(early_stopping=True).fit([[1.0], [2.0]], [0, 1])', /early_stopping/],
    ['from sklearn.linear_model import LogisticRegression', "LogisticRegression(solver='liblinear').fit([[1.0], [2.0]], [0, 1])", /liblinear/],
    ['from sklearn.linear_model import LogisticRegression', 'LogisticRegression(max_iter=5).fit([[1.0], [2.0]], [0, 1])', /max_iter=5/],
    ['from sklearn.linear_model import LogisticRegression', "LogisticRegression(penalty='l1').fit([[1.0], [2.0]], [0, 1])", /penalty='l1'/],
    ['from sklearn.linear_model import Ridge', "Ridge(solver='sag').fit([[1.0], [2.0]], [0, 1])", /solver='sag'/],
    ['from sklearn.tree import DecisionTreeClassifier', "DecisionTreeClassifier(max_features='sqrt').fit([[1.0], [2.0]], [0, 1])", /max_features='sqrt'/],
    ['from sklearn.tree import DecisionTreeClassifier', 'DecisionTreeClassifier(ccp_alpha=0.1).fit([[1.0], [2.0]], [0, 1])', /ccp_alpha/],
    ['from sklearn.decomposition import PCA', "PCA(svd_solver='randomized').fit([[1.0, 2.0], [2.0, 1.0]])", /randomized/],
    ['from sklearn.decomposition import PCA', "PCA(n_components='mle').fit([[1.0, 2.0], [2.0, 1.0]])", /mle/],
    ['from sklearn.preprocessing import OneHotEncoder', 'OneHotEncoder(min_frequency=2).fit([[1], [2]])', /min_frequency/],
    ['from sklearn.preprocessing import OneHotEncoder', 'OneHotEncoder(dtype=int).fit([[1], [2]]).transform([[1]])', /float64/],
    ['from sklearn.preprocessing import PolynomialFeatures', "PolynomialFeatures(order='F').fit([[1.0]])", /order='F'/],
    ['from sklearn.preprocessing import StandardScaler', "StandardScaler().fit([[1.0], [float('nan')]])", /NaN/],
    ['from sklearn.preprocessing import StandardScaler', 'StandardScaler().fit([[1.0]], sample_weight=[1])', /sample_weight/],
    ['from sklearn.naive_bayes import GaussianNB', 'GaussianNB().fit([[1.0], [2.0]], [0, 1], sample_weight=[1, 2])', /sample_weight/],
    ['from sklearn.metrics import accuracy_score', 'accuracy_score([0, 1], [0, 1], sample_weight=[1, 2])', /sample_weight/],
    ['from sklearn.metrics import r2_score', "r2_score([0.5, 1], [0, 1], multioutput='raw_values')", /multioutput/],
    ['from sklearn.datasets import load_digits', 'load_digits()', /only load_iris/],
  ]
  for (const [imp, src, message] of cases) {
    it(src.slice(0, 60), () => {
      const r = run(`${H}${imp}\n${src}`)
      expect(r.error?.type).toBe('NotImplementedError')
      expect(r.error?.message).toMatch(message)
    })
  }
})

describe('without a seed', () => {
  it('np.random.seed makes a run repeatable, and sizes and contents hold', () => {
    const src = `${H}from sklearn.model_selection import train_test_split\nnp.random.seed(0)\na, b = train_test_split(list(range(10)), test_size=0.3)\nprint(len(a), len(b), sorted(a + b))\nprint(a, b)`
    const one = run(src)
    expect(one.error).toBeNull()
    expect(one.stdout.split('\n')[0]).toBe('7 3 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]')
    expect(run(src).stdout).toBe(one.stdout)
  })
  it('a RandomState instance is taken as a source of seeds', () => {
    const r = run(`${H}from sklearn.cluster import KMeans\nm = KMeans(2, random_state=np.random.RandomState(0)).fit([[0.0], [0.1], [5.0], [5.1]])\nprint(sorted(m.labels_.tolist()))`)
    expect(r.error).toBeNull()
    expect(r.stdout).toBe('[0, 0, 1, 1]\n')
  })
})

describe('the step-through panel sees each call', () => {
  it('functions and estimator methods', () => {
    const r = run(`${H}from sklearn.model_selection import train_test_split, KFold\nfrom sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.metrics import accuracy_score\nX = np.array([[0.0], [1.0], [2.0], [3.0]])\ny = np.array([0, 0, 1, 1])\nA, B, c, d = train_test_split(X, y, random_state=0)\nm = KNeighborsClassifier(1)\nm.fit(A, c)\np = m.predict(B)\ns = accuracy_score(d, p)\nk = KFold(2).split(X)`)
    expect(r.error).toBeNull()
    expect(r.calls.filter((c) => c.api.startsWith('sklearn.')).map((c) => c.api)).toEqual(['sklearn.train_test_split', 'sklearn.KNeighborsClassifier.fit', 'sklearn.KNeighborsClassifier.predict', 'sklearn.accuracy_score', 'sklearn.KFold.split'])
    const pred = r.calls.find((c) => c.api === 'sklearn.KNeighborsClassifier.predict')!
    expect(pred.operands[0].label).toBe('B')
    expect(pred.result?.shape).toEqual([1])
  })
})
