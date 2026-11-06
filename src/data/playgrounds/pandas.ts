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
  // ---- the everyday APIs; each one was run on the real library
  {
    id: 'inspect', cat: 'basics', label: 'head / info / shape / dtypes',
    code: `import pandas as pd
import numpy as np

df = pd.DataFrame({'name': ['ann', 'bob', 'cat', 'dan'], 'age': [20, 31, 25, 40], 'score': [88.5, 72., np.nan, 91.]})
print(df.head(2))
print(df.tail(1))
print(df.shape, df.ndim, df.size, len(df), df.empty, df.columns.tolist(), df.index.tolist())
print(df.dtypes)
df.info()
print(df.nunique())
print(df['age'].value_counts().head(1), df['name'].unique(), df.values[0])`,
  },
  {
    id: 'sorting', cat: 'select', label: 'sort_values / nlargest / rank',
    code: `import pandas as pd
import numpy as np

df = pd.DataFrame({'name': ['ann', 'bob', 'cat', 'dan'], 'age': [20, 31, 25, 31], 'score': [88.5, 72., 60., 91.]})
print(df.sort_values('score', ascending=False).head(2))
print(df.sort_values(['age', 'score'], ascending=[False, True]))
print(df.sort_index(ascending=False).index.tolist())
print(df.nlargest(2, 'score')['name'].tolist(), df.nsmallest(1, 'age')['name'].tolist())
print(df['score'].rank(ascending=False).tolist(), df['score'].idxmax(), df['score'].idxmin())
print(df.reset_index(drop=True).shape, df.set_index('name').loc['bob', 'age'], df.set_index('name').reset_index().columns.tolist())`,
  },
  {
    id: 'col_ops', cat: 'select', label: 'assign / rename / drop / insert',
    code: `import pandas as pd
import numpy as np

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4., 5., 6.]})
df['c'] = df['a'] + df['b']
print(df.assign(d=lambda x: x.a * 2, e=0).columns.tolist())
print(df.rename(columns={'a': 'A'}).columns.tolist(), df.drop(columns=['c']).columns.tolist(), df.drop(index=[0]).index.tolist())
df.insert(0, 'id', [7, 8, 9])
print(df.pop('c').tolist(), df.columns.tolist())
print(df[['id', 'a']].shape, df.select_dtypes('number').columns.tolist(), df.select_dtypes(include='float').columns.tolist())
print(df.astype({'a': 'float64'}).dtypes['a'], df.a.astype(str).tolist(), df.add_prefix('x_').columns.tolist(), df.T.shape)
print(df.copy().equals(df), df.iat[0, 1], df.at[0, 'a'], df.get('nope', 'default'), df.keys().tolist())`,
  },
  {
    id: 'query', cat: 'select', label: 'query / isin / between',
    code: `import pandas as pd
import numpy as np

df = pd.DataFrame({'city': ['HK', 'SZ', 'HK', 'GZ'], 'age': [20, 31, 25, 40], 'n': [1, 2, 3, 4]})
print(df.query('age > 22 and city == "HK"'))
limit = 30
print(df.query('age > @limit')['city'].tolist())
print(df[df.city.isin(['SZ', 'GZ'])].n.tolist(), df[df.age.between(20, 25)].n.tolist(), df[~df.city.eq('HK')].n.tolist())
print(df[(df.age > 22) & (df.n < 4)].n.tolist(), df[(df.age < 21) | (df.n == 4)].n.tolist())
print(df.where(df.n > 2)['n'].tolist(), df.n.mask(df.n > 2, 0).tolist(), df.n.clip(2, 3).tolist())
print(df.sample(2, random_state=0).shape, df.duplicated('city').tolist(), df.drop_duplicates('city').n.tolist())`,
  },
  {
    id: 'stats', cat: 'basics', label: 'mean / corr / quantile / apply',
    code: `import pandas as pd
import numpy as np

df = pd.DataFrame({'a': [1., 2., 3., 4.], 'b': [2., 4., 5., 9.]})
print(df.sum().tolist(), df.mean().tolist(), df.median().tolist(), df.min().tolist(), df.max().tolist())
print(df.std().round(3).tolist(), df.var().round(3).tolist(), df.count().tolist(), df.a.mode().tolist(), df.prod().tolist())
print(df.mean(axis=1).tolist(), df.a.quantile(0.5), df.quantile([0.25, 0.75]).shape, df.skew().round(3).tolist(), df.kurt().round(3).tolist())
print(df.corr().round(3))
print(df.cov().round(3), df.a.corr(df.b).round(3))
print(df.apply(np.sum).tolist(), df.apply(lambda r: r.a + r.b, axis=1).tolist(), df.a.apply(lambda v: v * 2).tolist(), df.a.map({1.: 'x'}).tolist())
print(df.cumsum().iloc[-1].tolist(), df.a.diff().tolist(), df.a.pct_change().round(3).tolist(), df.a.shift(1).tolist(), df.abs().sum().tolist(), df.round(0).shape)`,
  },
  {
    id: 'groupby_more', cat: 'group', label: 'groupby: size / first / cumcount',
    code: `import pandas as pd
import numpy as np

df = pd.DataFrame({'t': ['a', 'b', 'a', 'b', 'a'], 'x': [1, 2, 3, 4, 5], 'y': [5., 4., 3., 2., 1.]})
g = df.groupby('t')
print(g.size())
print(g.ngroups, sorted(g.groups), g.get_group('a').x.tolist(), g.x.sum().tolist(), g.x.mean().tolist(), g.x.count().tolist())
print(g.first(), g.last().x.tolist(), g.head(1).x.tolist(), g.nth(0).x.tolist())
print(g.x.max().tolist(), g.x.min().tolist(), g.x.median().tolist(), g.x.std().round(3).tolist(), g.x.var().tolist(), g.x.nunique().tolist())
print(g.x.cumsum().tolist(), g.cumcount().tolist(), g.x.shift(1).tolist(), g.x.rank().tolist(), g.x.idxmax().tolist())
print(df.groupby('t', as_index=False).x.sum())
print(df.groupby('t').filter(lambda d: len(d) > 2).x.tolist())
for name, part in g:
    print(name, len(part))`,
  },
  {
    id: 'pivot_more', cat: 'combine', label: 'pivot_table / stack / join',
    code: `import pandas as pd
import numpy as np

df = pd.DataFrame({'y': [1, 1, 2, 2], 'k': ['a', 'b', 'a', 'b'], 'v': [1., 2., 3., 4.]})
print(df.pivot(index='y', columns='k', values='v'))
print(df.pivot_table(index='k', values='v', aggfunc='mean'))
print(df.pivot_table(index='y', columns='k', values='v', aggfunc='sum', margins=True))
w = df.pivot(index='y', columns='k', values='v')
print(w.stack().tolist(), w.stack().unstack().shape, w.reset_index().melt(id_vars='y').shape)
a = pd.DataFrame({'x': [1, 2]}, index=['p', 'q'])
b = pd.DataFrame({'z': [3, 4]}, index=['q', 'r'])
print(a.join(b), a.join(b, how='outer'), a.combine_first(pd.DataFrame({'x': [9]}, index=['z'])).x.tolist())
print(pd.concat([a, b], axis=1).shape, a.reindex(['q', 'p', 'n']).x.tolist(), a.rename_axis('id').index.name)`,
  },
  {
    id: 'clean_more', cat: 'clean', label: 'replace / to_numeric / ffill / cut',
    code: `import pandas as pd
import numpy as np

s = pd.Series([1, 2, 3, 4, 5, 6])
print(s.replace(1, 9).tolist(), s.replace({2: 8}).tolist())
print(pd.to_numeric(pd.Series(['1', '2.5', 'x']), errors='coerce').tolist())
m = pd.Series([1., np.nan, np.nan, 4.])
print(m.ffill().tolist(), m.bfill().tolist(), m.interpolate().tolist(), m.fillna(m.mean()).tolist(), m.isna().sum(), m.notna().tolist(), m.dropna().tolist(), m.hasnans)
print(pd.cut(s, 2, labels=['low', 'high']).tolist(), pd.cut(s, [0, 2, 6]).value_counts().tolist(), pd.qcut(s, 2, labels=False).tolist())
print(s.astype('category').cat.categories.tolist(), pd.factorize(['b', 'a', 'b'])[0].tolist(), pd.Series(pd.Categorical(['a', 'b', 'a'])).cat.codes.tolist(), pd.unique(pd.Series([1, 1])).tolist())
df = pd.DataFrame({'a': [1, np.nan], 'b': [np.nan, np.nan]})
print(df.isna().sum().tolist(), df.dropna(how='all').shape, df.dropna(axis=1, how='all').shape, df.fillna({'a': 0}).a.tolist(), df.notna().sum().sum(), df.isnull().any().tolist())`,
  },
  {
    id: 'str_more', cat: 'series', label: 'str.contains / extract / split',
    code: `import pandas as pd
import numpy as np

s = pd.Series(['Ann Lee', 'bob wu', None, 'Cat Ho 12'])
print(s.str.lower().tolist(), s.str.upper().tolist(), s.str.title().tolist(), s.str.len().tolist(), s.str.strip().tolist())
print(s.str.contains('a', na=False).tolist(), s.str.startswith('A', na=False).tolist(), s.str.endswith('u', na=False).tolist(), s.str.match('[A-Z]', na=False).tolist())
print(s.str.replace(' ', '_').tolist(), s.str.split().str[0].tolist(), s.str.split(' ', expand=True).shape, s.str[:2].tolist(), s.str.slice(0, 1).tolist(), s.str.get(0).tolist())
print(s.str.extract(r'(\\d+)')[0].tolist(), s.str.findall(r'[A-Z]').tolist(), s.str.count('a').tolist(), s.str.find('b').tolist(), s.str.pad(3).str.len().tolist(), s.str.zfill(8).tolist())
print(s.str.cat(sep='|'), s.str.isdigit().tolist(), pd.Series(['7']).str.isnumeric().tolist(), s.str.capitalize().tolist(), s.str.get_dummies(' ').shape)`,
  },
  {
    id: 'output', cat: 'basics', label: 'to_csv / to_dict / iterrows',
    code: `import pandas as pd
import numpy as np

df = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
print(df.to_csv(index=False))
df.to_csv('t.csv', index=False)
print(pd.read_csv('t.csv').shape, pd.read_table('t.csv', sep=',').shape)
print(df.to_dict(), df.to_dict('records'), df.to_dict('list'))
print(df.to_json(), df.to_numpy().shape, df.a.to_numpy(), df.a.to_list(), df.a.to_frame().shape, df.to_string(index=False))
for i, row in df.iterrows():
    print(i, row['a'], row.b)
for t in df.itertuples():
    print(t.Index, t.a, t.b)
print(pd.DataFrame.from_dict({'a': [1]}), pd.DataFrame.from_records([{'a': 1}]).shape, pd.DataFrame([[1, 2]], columns=['x', 'y']).shape, pd.Series({'p': 1}).index.tolist(), pd.read_json('[{"a": 1}]').shape)`,
  },
  {
    id: 'plot_like', cat: 'ml', label: 'train/test by hand: sample + drop',
    code: `import pandas as pd
import numpy as np

df = pd.DataFrame({'x': range(10), 'y': [0, 1] * 5})
train = df.sample(frac=0.8, random_state=0)
test = df.drop(train.index)
print(len(train), len(test), set(train.index) & set(test.index))
X, y = df.drop(columns='y').to_numpy(), df['y'].to_numpy()
print(X.shape, y.shape, X.dtype, y.mean())
norm = (df.x - df.x.mean()) / df.x.std()
print(norm.round(2).tolist()[:3], ((df.x - df.x.min()) / (df.x.max() - df.x.min())).tolist()[:2])
print(pd.get_dummies(pd.Series(['a', 'b']), dtype=int).to_numpy().tolist(), df.y.value_counts(normalize=True).tolist(), df.y.nunique(), pd.crosstab(df.y, df.x > 4).shape)`,
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
