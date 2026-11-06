// pandas 实验台：读表、选取、清洗，以及课程里围绕模型做的表格计算（朴素贝叶斯计数、标准化、K-Means 质心、混淆矩阵）
import { PANDAS } from '../../lib/pyPandas'
import { MATPLOTLIB } from '../../lib/pyPlot'
import type { PlayConfig, PlayEntry } from './types'

const H = 'import pandas as pd\nimport numpy as np\n\n'

/** lecture 2's table: 14 patients, four symptoms, disease Z */
const DISEASE = `df = pd.DataFrame({
    'BloodPressure': ['High', 'High', 'Low', 'Normal', 'Normal', 'Normal', 'Low', 'High', 'High', 'Normal', 'High', 'Low', 'Low', 'Normal'],
    'Fever':    ['High', 'High', 'High', 'Mild', 'No fever', 'No fever', 'No fever', 'Mild', 'No fever', 'Mild', 'Mild', 'Mild', 'High', 'Mild'],
    'Diabetes': ['Yes', 'Yes', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'Yes'],
    'Vomit':    ['No', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes'],
    'DiseaseZ': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No'],
})
`

const PEOPLE = `df = pd.DataFrame({
    'name':   ['Ann', 'Bob', 'Cy', 'Dee', 'Eve'],
    'height': [160, 175, 168, 182, 155],
    'weight': [52.5, 70.0, None, 81.2, 48.0],
    'label':  ['M', 'L', 'M', 'L', 'M'],
})
`

const ENTRIES: PlayEntry[] = [
  // ---- basics
  {
    id: 'create', cat: 'basics', label: 'pd.DataFrame({...})',
    code: `${H}${PEOPLE}print(df)
print(df.shape)          # (rows, columns)
print(df.dtypes)         # int64, float64 (None -> NaN), object (text)
df.head(3)`,
  },
  {
    id: 'describe', cat: 'basics', label: 'describe() / mean()', focus: ['describe'],
    code: `${H}${PEOPLE}print(df[['height', 'weight']].mean())     # NaN is skipped
print(df['height'].std())                   # pandas: ddof=1
print(np.std(df['height'].to_numpy()))      # NumPy: ddof=0
df.describe()`,
  },
  {
    id: 'csv', cat: 'basics', label: 'pd.read_csv',
    code: `${H}import io

text = """height,weight,label
160,52.5,M
175,70.0,L
168,,M
182,81.2,L"""
df = pd.read_csv(io.StringIO(text))   # in Colab: pd.read_csv('data.csv')
print(df)
print(df.dtypes)
df.info()`,
  },

  // ---- select
  {
    id: 'columns', cat: 'select', label: "df['col'] / df[['a', 'b']]",
    code: `${H}${PEOPLE}heights = df['height']              # one column: a Series
print(type(heights).__name__)
X = df[['height', 'weight']]         # a list of columns: a DataFrame
print(X)
y = df['label']
print(y.tolist())`,
  },
  {
    id: 'filter', cat: 'select', label: 'df[df.height > 170]', focus: ['frame'],
    code: `${H}${PEOPLE}tall = df[df['height'] > 170]
print(tall)
# combine conditions with & and |, each in brackets
df[(df['height'] > 160) & (df['label'] == 'M')]`,
  },
  {
    id: 'loc', cat: 'select', label: 'loc vs iloc',
    code: `${H}${PEOPLE}df = df.set_index('name')
print(df.loc['Bob':'Dee', 'height'])   # labels: the end is INCLUDED
print(df.iloc[1:3, 0])                 # positions: the end is excluded
df.loc[df['label'] == 'L', ['height', 'weight']]`,
  },

  // ---- clean
  {
    id: 'missing', cat: 'clean', label: 'isna / fillna / dropna',
    code: `${H}${PEOPLE}print(df.isna().sum())                         # missing values per column
filled = df.fillna({'weight': df['weight'].mean()})
print(filled)
df.dropna()`,
  },
  {
    id: 'encode', cat: 'clean', label: 'map / get_dummies',
    code: `${H}${PEOPLE}df['y'] = df['label'].map({'M': 0, 'L': 1})     # labels -> numbers
print(df[['label', 'y']])
pd.get_dummies(df['label'])                      # one-hot columns`,
  },

  // ---- course calculations
  {
    id: 'naive_bayes', cat: 'ml', label: 'Naive Bayes counts (lecture 2)', focus: ['crosstab'], focusFirst: true,
    code: `${H}${DISEASE}
prior = df['DiseaseZ'].value_counts(normalize=True)                 # P(Z)
print(prior)
bp = pd.crosstab(df['DiseaseZ'], df['BloodPressure'], normalize='index')  # P(BloodPressure | Z)
print(bp)

# a patient with High blood pressure, no fever, diabetes and vomiting
x = {'BloodPressure': 'High', 'Fever': 'No fever', 'Diabetes': 'Yes', 'Vomit': 'Yes'}
for z in ['Yes', 'No']:
    p = prior[z]
    for feature, value in x.items():
        table = pd.crosstab(df['DiseaseZ'], df[feature], normalize='index')
        p *= table.loc[z, value]
    print(z, round(p, 5))`,
  },
  {
    id: 'standardize', cat: 'ml', label: 'z-score for KNN', focus: ['arith'],
    code: `${H}df = pd.DataFrame({'height': [160, 175, 168, 182, 155],
                   'weight': [52.5, 70.0, 64.1, 81.2, 48.0]})
z = (df - df.mean()) / df.std(ddof=0)      # ddof=0, as np.std does
print(z.round(3))
print(z.mean().round(6).tolist(), z.std(ddof=0).tolist())`,
  },
  {
    id: 'centroids', cat: 'ml', label: 'K-Means: groupby().mean()', focus: ['groupby'], focusFirst: true,
    code: `${H}points = pd.DataFrame({'x': [1, 1.5, 3, 5, 3.5, 4.5, 3.5],
                       'y': [1, 2, 4, 7, 5, 5, 4.5],
                       'cluster': [0, 0, 1, 1, 1, 1, 1]})
# K-Means step 4: each centroid moves to the mean of its points
centroids = points.groupby('cluster')[['x', 'y']].mean()
print(centroids)
points.groupby('cluster').size()`,
  },
  {
    id: 'confusion', cat: 'ml', label: 'crosstab = confusion matrix', focus: ['crosstab'],
    code: `${H}results = pd.DataFrame({'true': [1, 0, 1, 1, 0, 1, 0, 0, 1, 1],
                        'pred': [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]})
cm = pd.crosstab(results['true'], results['pred'])
print(cm)
tp = cm.loc[1, 1]
precision = tp / cm[1].sum()          # column: everything predicted 1
recall = tp / cm.loc[1].sum()         # row: everything truly 1
print(f"precision {precision:.2f}, recall {recall:.2f}")`,
  },
  {
    id: 'groupby', cat: 'ml', label: 'groupby + plot',
    code: `${H}import matplotlib.pyplot as plt
scores = pd.DataFrame({'k': [1, 1, 3, 3, 5, 5, 7, 7],
                       'fold': [1, 2, 1, 2, 1, 2, 1, 2],
                       'accuracy': [0.80, 0.84, 0.88, 0.86, 0.91, 0.89, 0.90, 0.88]})
mean_acc = scores.groupby('k')['accuracy'].mean()     # average over folds
print(mean_acc)
best_k = mean_acc.idxmax()
plt.bar([str(k) for k in mean_acc.index], mean_acc.tolist())
plt.ylim(0.75, 0.95)
plt.title(f'cross-validation: best K = {best_k}')`,
  },
  {
    id: 'isin', cat: 'select', label: ".isin([...])",
    code: `${H}df = pd.DataFrame({'student': ['Ann', 'Bob', 'Cy', 'Dee', 'Eve'],
                   'major': ['COMP', 'MATH', 'COMP', 'PHYS', 'ELEC']})
df[df['major'].isin(['COMP', 'ELEC'])]`,
  },
  {
    id: 'nlargest', cat: 'select', label: "df.nlargest(3, col)",
    code: `${H}df = pd.DataFrame({'model': ['KNN', 'NB', 'MLP', 'CNN', 'Perceptron'],
                   'accuracy': [0.88, 0.81, 0.91, 0.95, 0.76]})
df.nlargest(3, 'accuracy')`,
  },
  {
    id: 'duplicates', cat: 'clean', label: "duplicated / drop_duplicates",
    code: `${H}df = pd.DataFrame({'name': ['Ann', 'Bob', 'Ann', 'Cy', 'Bob'],
                   'score': [90, 75, 90, 60, 80]})
print(df.duplicated())        # True = an exact repeat of an earlier row
df.drop_duplicates()`,
  },
  {
    id: 'apply_rows', cat: 'clean', label: "apply(…, axis=1)",
    code: `${H}df = pd.DataFrame({'height': [170, 160, 182], 'weight': [65, 50, 90]})
df['bmi'] = df.apply(lambda r: r.weight / (r.height / 100) ** 2, axis=1)
df.round(1)`,
  },
  {
    id: 'cut', cat: 'clean', label: "pd.cut → categories",
    code: `${H}df = pd.DataFrame({'age': [22, 35, 58, 41, 19, 66, 30, 47],
                   'buys': ['yes', 'no', 'no', 'yes', 'yes', 'no', 'yes', 'no']})
df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 100], labels=['young', 'middle', 'old'])
pd.crosstab(df['age_group'], df['buys'])     # the counts Naive Bayes needs`,
  },
  {
    id: 'gaussian_nb', cat: 'ml', label: "agg(['mean', 'std'])",
    code: `${H}df = pd.DataFrame({'cls': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham'],
                   'length': [12.0, 30.0, 9.0, 26.0, 15.0, 34.0]})
params = df.groupby('cls')['length'].agg(['mean', 'std'])
params      # Gaussian Naive Bayes: one mean and one std per class`,
  },
  {
    id: 'pivot', cat: 'ml', label: "pivot_table",
    code: `${H}df = pd.DataFrame({'cluster': [0, 1, 0, 1, 0, 1],
                   'x': [1.0, 8.0, 2.0, 9.0, 1.5, 7.0],
                   'y': [2.0, 8.5, 1.0, 9.5, 1.5, 8.0]})
df.pivot_table(index='cluster', values=['x', 'y'], aggfunc='mean')   # the new K-Means centroids`,
  },
  {
    id: 'sample_split', cat: 'ml', label: "sample → train / test",
    code: `${H}df = pd.DataFrame({'x': range(8), 'label': [0, 1, 0, 1, 1, 0, 1, 0]})
train = df.sample(frac=0.75, random_state=0)
test = df.drop(train.index)     # every row not drawn for training
print(len(train), 'train rows,', len(test), 'test rows')
test`,
  },
  // ---- what the sandbox gained later; each one was run on the real library
  {
    id: 'merge', cat: 'combine', label: 'pd.merge',
    code: `import pandas as pd
import numpy as np

students = pd.DataFrame({'id': [1, 2, 3], 'name': ['Ann', 'Bo', 'Cy']})
marks = pd.DataFrame({'id': [1, 2, 4], 'mark': [90, 75, 60]})
print(pd.merge(students, marks, on='id'))
print(pd.merge(students, marks, on='id', how='left'))`,
  },
  {
    id: 'concat', cat: 'combine', label: 'pd.concat',
    code: `import pandas as pd
import numpy as np

a = pd.DataFrame({'x': [1, 2]})
b = pd.DataFrame({'x': [3], 'y': [4]})
print(pd.concat([a, b]))
print(pd.concat([a, b], ignore_index=True))
print(pd.concat([a, a], axis=1))`,
  },
  {
    id: 'melt', cat: 'combine', label: 'df.melt',
    code: `import pandas as pd
import numpy as np

wide = pd.DataFrame({'name': ['Ann', 'Bo'], 'math': [90, 80], 'art': [70, 95]})
long = wide.melt(id_vars='name', var_name='subject', value_name='mark')
print(long)
print(long.pivot(index='name', columns='subject', values='mark'))`,
  },
  {
    id: 'crosstab', cat: 'combine', label: 'pd.crosstab',
    code: `import pandas as pd
import numpy as np

df = pd.DataFrame({'pred': [1, 0, 1, 1, 0], 'true': [1, 0, 0, 1, 1]})
print(pd.crosstab(df['true'], df['pred']))
print(pd.crosstab(df['true'], df['pred'], normalize='index'))`,
  },
  {
    id: 'agg', cat: 'group', label: 'groupby.agg',
    code: `import pandas as pd
import numpy as np

df = pd.DataFrame({'cls': ['a', 'a', 'b', 'b'], 'x': [1, 2, 3, 5], 'y': [10, 20, 30, 40]})
print(df.groupby('cls').agg({'x': 'mean', 'y': ['min', 'max']}))
print(df.groupby('cls')['x'].agg(['mean', 'std', 'count']))`,
  },
  {
    id: 'transform', cat: 'group', label: 'groupby.transform',
    code: `import pandas as pd
import numpy as np

df = pd.DataFrame({'cls': ['a', 'a', 'b', 'b'], 'x': [1., 3., 10., 30.]})
df['mean'] = df.groupby('cls')['x'].transform('mean')
df['centred'] = df['x'] - df['mean']
print(df)
print(df.groupby('cls')['x'].rank())`,
  },
  {
    id: 'strings', cat: 'series', label: 'Series.str',
    code: `import pandas as pd
import numpy as np

s = pd.Series(['  Cat ', 'dog', None, 'Bird 2'])
print(s.str.strip().str.lower())
print(s.str.contains('o', na=False))
print(s.str.len())
print(s.str.split().str[0])`,
  },
  {
    id: 'rolling', cat: 'series', label: 'rolling / cumsum',
    code: `import pandas as pd
import numpy as np

s = pd.Series([1, 2, 3, 4, 5, 6])
print(s.rolling(3).mean())
print(s.expanding().max())
print(s.cumsum(), s.pct_change().round(2), s.shift(1))`,
  },
  {
    id: 'dummies', cat: 'series', label: 'pd.get_dummies',
    code: `import pandas as pd
import numpy as np

df = pd.DataFrame({'colour': ['red', 'blue', 'red'], 'size': [1, 2, 3]})
print(pd.get_dummies(df, columns=['colour']))
print(df['colour'].value_counts())
print(df['colour'].map({'red': 0, 'blue': 1}))`,
  },
]

export const PANDAS_PLAYGROUND: PlayConfig = {
  id: 'pandas',
  libs: [PANDAS, MATPLOTLIB],
  maxSize: 2_000_000,
  timeBudget: 2000,
  debounce: 400,
  cats: ['basics', 'select', 'clean', 'ml', 'combine', 'group', 'series'],
  entries: ENTRIES,
  prelude: H,
  callApis: [],
}
