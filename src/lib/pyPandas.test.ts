import { describe, expect, it } from 'vitest'
import { runPython } from './minipy'
import { PANDAS } from './pyPandas'

const run = (src: string) => runPython(src, { libs: [PANDAS], maxSize: 65536 })

// Expected stdout produced by pandas 2.3 on the same source.
const CASES: Record<string, { src: string; stdout: string }> = {
  // 2026-10-08, pandas 2.3.3: cut (labels / int bins / edges), categorical crosstab and groupby, duplicates, isin, nlargest,
  // groupby agg list, pivot_table (with and without columns), apply(axis=1) with row attributes, quantile, dtypes
  "cut_dupes_pivot": {
    "src": "import pandas as pd\nimport numpy as np\ndf = pd.DataFrame({'age': [22, 35, 58, 41, 19, 66], 'buys': ['yes', 'no', 'no', 'yes', 'yes', 'no']})\ndf['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 100], labels=['young', 'middle', 'old'])\nprint(repr(df['age_group']))\nprint(repr(pd.crosstab(df['age_group'], df['buys'])))\nprint(df['age_group'].dtype)\nprint(repr(pd.cut(pd.Series([1, 2, 3, 10]), bins=3)))\nprint(repr(pd.cut(pd.Series([1, 5, 9]), bins=[0, 4, 8, 12])))\nprint(repr(df))\nd2 = pd.DataFrame({'name': ['Ann', 'Bob', 'Ann', 'Cy', 'Bob'], 'score': [90, 75, 90, 60, 80]})\nprint(repr(d2.duplicated()))\nprint(repr(d2.drop_duplicates()))\nprint(repr(d2.drop_duplicates(subset=['name'])))\nprint(repr(d2[d2['name'].isin(['Ann', 'Cy'])]))\nprint(repr(d2.nlargest(3, 'score')))\nprint(repr(d2['score'].nsmallest(2)))\ng = pd.DataFrame({'cls': ['a', 'b', 'a', 'b', 'a'], 'x': [1.0, 4.0, 3.0, 6.0, 2.0]})\nprint(repr(g.groupby('cls')['x'].agg(['mean', 'std', 'count'])))\nprint(repr(g.pivot_table(index='cls', values='x', aggfunc='mean')))\nh = pd.DataFrame({'height': [170, 160], 'weight': [65, 50]})\nprint(repr(h.apply(lambda r: r.weight / (r.height / 100) ** 2, axis=1)))\nprint(g['x'].quantile(0.5), g['x'].quantile(0.25))\np2 = pd.DataFrame({'cls': ['a', 'b', 'a', 'b'], 'f': ['u', 'u', 'v', 'u'], 'x': [1, 2, 3, 4]})\nprint(repr(p2.pivot_table(index='cls', columns='f', values='x', aggfunc='sum')))\nprint(repr(df.groupby('age_group')['age'].mean()))\nprint(df.dtypes)\n",
    "stdout": "0     young\n1    middle\n2       old\n3    middle\n4     young\n5       old\nName: age_group, dtype: category\nCategories (3, object): ['young' < 'middle' < 'old']\nbuys       no  yes\nage_group         \nyoung       0    2\nmiddle      1    1\nold         2    0\ncategory\n0    (0.991, 4.0]\n1    (0.991, 4.0]\n2    (0.991, 4.0]\n3     (7.0, 10.0]\ndtype: category\nCategories (3, interval[float64, right]): [(0.991, 4.0] < (4.0, 7.0] < (7.0, 10.0]]\n0     (0, 4]\n1     (4, 8]\n2    (8, 12]\ndtype: category\nCategories (3, interval[int64, right]): [(0, 4] < (4, 8] < (8, 12]]\n   age buys age_group\n0   22  yes     young\n1   35   no    middle\n2   58   no       old\n3   41  yes    middle\n4   19  yes     young\n5   66   no       old\n0    False\n1    False\n2     True\n3    False\n4    False\ndtype: bool\n  name  score\n0  Ann     90\n1  Bob     75\n3   Cy     60\n4  Bob     80\n  name  score\n0  Ann     90\n1  Bob     75\n3   Cy     60\n  name  score\n0  Ann     90\n2  Ann     90\n3   Cy     60\n  name  score\n0  Ann     90\n2  Ann     90\n4  Bob     80\n3    60\n1    75\nName: score, dtype: int64\n     mean       std  count\ncls                       \na     2.0  1.000000      3\nb     5.0  1.414214      2\n       x\ncls     \na    2.0\nb    5.0\n0    22.491349\n1    19.531250\ndtype: float64\n3.0 2.0\nf      u    v\ncls          \na    1.0  3.0\nb    6.0  NaN\nage_group\nyoung     20.5\nmiddle    38.0\nold       62.0\nName: age, dtype: float64\nage             int64\nbuys           object\nage_group    category\ndtype: object\n"
  },
  "print_df": {
    "src": "import pandas as pd\nimport numpy as np\ndf = pd.DataFrame({'name': ['Ann', 'Bob', 'Cy', 'Dee'], 'age': [23, 31, 19, 45], 'score': [88.5, 92.25, None, 70.0], 'passed': [True, True, False, True]})\nd2 = pd.DataFrame({'label': ['spam', 'ham', 'spam', 'ham', 'ham'], 'free': [1, 0, 1, 0, 1], 'len': [10, 25, 12, 30, 8]})\nprint(df)\nprint(df.dtypes)\nprint(df.shape, list(df.columns))",
    "stdout": "  name  age  score  passed\n0  Ann   23  88.50    True\n1  Bob   31  92.25    True\n2   Cy   19    NaN   False\n3  Dee   45  70.00    True\nname       object\nage         int64\nscore     float64\npassed       bool\ndtype: object\n(4, 4) ['name', 'age', 'score', 'passed']\n"
  },
  "select": {
    "src": "import pandas as pd\nimport numpy as np\ndf = pd.DataFrame({'name': ['Ann', 'Bob', 'Cy', 'Dee'], 'age': [23, 31, 19, 45], 'score': [88.5, 92.25, None, 70.0], 'passed': [True, True, False, True]})\nd2 = pd.DataFrame({'label': ['spam', 'ham', 'spam', 'ham', 'ham'], 'free': [1, 0, 1, 0, 1], 'len': [10, 25, 12, 30, 8]})\nprint(df['age'])\nprint(df[['name', 'age']])\nprint(df[df['age'] > 20])",
    "stdout": "0    23\n1    31\n2    19\n3    45\nName: age, dtype: int64\n  name  age\n0  Ann   23\n1  Bob   31\n2   Cy   19\n3  Dee   45\n  name  age  score  passed\n0  Ann   23  88.50    True\n1  Bob   31  92.25    True\n3  Dee   45  70.00    True\n"
  },
  "loc_iloc": {
    "src": "import pandas as pd\nimport numpy as np\ndf = pd.DataFrame({'name': ['Ann', 'Bob', 'Cy', 'Dee'], 'age': [23, 31, 19, 45], 'score': [88.5, 92.25, None, 70.0], 'passed': [True, True, False, True]})\nd2 = pd.DataFrame({'label': ['spam', 'ham', 'spam', 'ham', 'ham'], 'free': [1, 0, 1, 0, 1], 'len': [10, 25, 12, 30, 8]})\nprint(df.loc[1:2, 'name':'score'])\nprint(df.iloc[1:3, [0, 2]])\nprint(df.loc[0, 'name'], df.iloc[-1, 1])\nprint(df.loc[2])",
    "stdout": "  name  age  score\n1  Bob   31  92.25\n2   Cy   19    NaN\n  name  score\n1  Bob  92.25\n2   Cy    NaN\nAnn 45\nname         Cy\nage          19\nscore       NaN\npassed    False\nName: 2, dtype: object\n"
  },
  "describe": {
    "src": "import pandas as pd\nimport numpy as np\ndf = pd.DataFrame({'name': ['Ann', 'Bob', 'Cy', 'Dee'], 'age': [23, 31, 19, 45], 'score': [88.5, 92.25, None, 70.0], 'passed': [True, True, False, True]})\nd2 = pd.DataFrame({'label': ['spam', 'ham', 'spam', 'ham', 'ham'], 'free': [1, 0, 1, 0, 1], 'len': [10, 25, 12, 30, 8]})\nprint(df.describe())\nprint(df['score'].mean(), df['age'].std(), np.std(df['age'].to_numpy()))",
    "stdout": "            age      score\ncount   4.00000   3.000000\nmean   29.50000  83.583333\nstd    11.47461  11.912004\nmin    19.00000  70.000000\n25%    22.00000  79.250000\n50%    27.00000  88.500000\n75%    34.50000  90.375000\nmax    45.00000  92.250000\n83.58333333333333 11.474609652039003 9.937303457175895\n"
  },
  "clean": {
    "src": "import pandas as pd\nimport numpy as np\ndf = pd.DataFrame({'name': ['Ann', 'Bob', 'Cy', 'Dee'], 'age': [23, 31, 19, 45], 'score': [88.5, 92.25, None, 70.0], 'passed': [True, True, False, True]})\nd2 = pd.DataFrame({'label': ['spam', 'ham', 'spam', 'ham', 'ham'], 'free': [1, 0, 1, 0, 1], 'len': [10, 25, 12, 30, 8]})\nprint(df.isna())\nprint(df.fillna(0))\nprint(df.dropna())\nprint(df.sort_values('age', ascending=False))",
    "stdout": "    name    age  score  passed\n0  False  False  False   False\n1  False  False  False   False\n2  False  False   True   False\n3  False  False  False   False\n  name  age  score  passed\n0  Ann   23  88.50    True\n1  Bob   31  92.25    True\n2   Cy   19   0.00   False\n3  Dee   45  70.00    True\n  name  age  score  passed\n0  Ann   23  88.50    True\n1  Bob   31  92.25    True\n3  Dee   45  70.00    True\n  name  age  score  passed\n3  Dee   45  70.00    True\n1  Bob   31  92.25    True\n0  Ann   23  88.50    True\n2   Cy   19    NaN   False\n"
  },
  "counts": {
    "src": "import pandas as pd\nimport numpy as np\ndf = pd.DataFrame({'name': ['Ann', 'Bob', 'Cy', 'Dee'], 'age': [23, 31, 19, 45], 'score': [88.5, 92.25, None, 70.0], 'passed': [True, True, False, True]})\nd2 = pd.DataFrame({'label': ['spam', 'ham', 'spam', 'ham', 'ham'], 'free': [1, 0, 1, 0, 1], 'len': [10, 25, 12, 30, 8]})\nprint(d2['label'].value_counts())\nprint(d2['label'].value_counts(normalize=True))",
    "stdout": "label\nham     3\nspam    2\nName: count, dtype: int64\nlabel\nham     0.6\nspam    0.4\nName: proportion, dtype: float64\n"
  },
  "groupby": {
    "src": "import pandas as pd\nimport numpy as np\ndf = pd.DataFrame({'name': ['Ann', 'Bob', 'Cy', 'Dee'], 'age': [23, 31, 19, 45], 'score': [88.5, 92.25, None, 70.0], 'passed': [True, True, False, True]})\nd2 = pd.DataFrame({'label': ['spam', 'ham', 'spam', 'ham', 'ham'], 'free': [1, 0, 1, 0, 1], 'len': [10, 25, 12, 30, 8]})\nprint(d2.groupby('label')['len'].mean())\nprint(d2.groupby('label').mean())\nprint(d2.groupby('label').size())",
    "stdout": "label\nham     21.0\nspam    11.0\nName: len, dtype: float64\n           free   len\nlabel                \nham    0.333333  21.0\nspam   1.000000  11.0\nlabel\nham     3\nspam    2\ndtype: int64\n"
  },
  "crosstab": {
    "src": "import pandas as pd\nimport numpy as np\ndf = pd.DataFrame({'name': ['Ann', 'Bob', 'Cy', 'Dee'], 'age': [23, 31, 19, 45], 'score': [88.5, 92.25, None, 70.0], 'passed': [True, True, False, True]})\nd2 = pd.DataFrame({'label': ['spam', 'ham', 'spam', 'ham', 'ham'], 'free': [1, 0, 1, 0, 1], 'len': [10, 25, 12, 30, 8]})\nprint(pd.crosstab(d2['label'], d2['free']))\nprint(pd.crosstab(d2['label'], d2['free'], normalize='index'))",
    "stdout": "free   0  1\nlabel      \nham    2  1\nspam   0  2\nfree          0         1\nlabel                    \nham    0.666667  0.333333\nspam   0.000000  1.000000\n"
  },
  "dummies_z": {
    "src": "import pandas as pd\nimport numpy as np\ndf = pd.DataFrame({'name': ['Ann', 'Bob', 'Cy', 'Dee'], 'age': [23, 31, 19, 45], 'score': [88.5, 92.25, None, 70.0], 'passed': [True, True, False, True]})\nd2 = pd.DataFrame({'label': ['spam', 'ham', 'spam', 'ham', 'ham'], 'free': [1, 0, 1, 0, 1], 'len': [10, 25, 12, 30, 8]})\nprint(pd.get_dummies(d2['label']))\nprint((d2[['free', 'len']] - d2[['free', 'len']].mean()) / d2[['free', 'len']].std())\nprint(d2['len'].describe())",
    "stdout": "     ham   spam\n0  False   True\n1   True  False\n2  False   True\n3   True  False\n4   True  False\n       free       len\n0  0.730297 -0.710742\n1 -1.095445  0.812277\n2  0.730297 -0.507673\n3 -1.095445  1.319950\n4  0.730297 -0.913812\ncount     5.000000\nmean     17.000000\nstd       9.848858\nmin       8.000000\n25%      10.000000\n50%      12.000000\n75%      25.000000\nmax      30.000000\nName: len, dtype: float64\n"
  },
  "series": {
    "src": "import pandas as pd\nimport numpy as np\ndf = pd.DataFrame({'name': ['Ann', 'Bob', 'Cy', 'Dee'], 'age': [23, 31, 19, 45], 'score': [88.5, 92.25, None, 70.0], 'passed': [True, True, False, True]})\nd2 = pd.DataFrame({'label': ['spam', 'ham', 'spam', 'ham', 'ham'], 'free': [1, 0, 1, 0, 1], 'len': [10, 25, 12, 30, 8]})\ns = pd.Series([1.5, 2.25, 3.0], index=['a', 'b', 'c'], name='x')\nprint(s)\nprint(s * 2)\nprint(pd.Series([1, 2, 3]))\nprint(d2.head(2))",
    "stdout": "a    1.50\nb    2.25\nc    3.00\nName: x, dtype: float64\na    3.0\nb    4.5\nc    6.0\nName: x, dtype: float64\n0    1\n1    2\n2    3\ndtype: int64\n  label  free  len\n0  spam     1   10\n1   ham     0   25\n"
  },
  "map_assign": {
    "src": "import pandas as pd\nimport numpy as np\ndf = pd.DataFrame({'name': ['Ann', 'Bob', 'Cy', 'Dee'], 'age': [23, 31, 19, 45], 'score': [88.5, 92.25, None, 70.0], 'passed': [True, True, False, True]})\nd2 = pd.DataFrame({'label': ['spam', 'ham', 'spam', 'ham', 'ham'], 'free': [1, 0, 1, 0, 1], 'len': [10, 25, 12, 30, 8]})\nprint(d2['label'].map({'spam': 1, 'ham': 0}))\nprint(d2.assign(ratio=d2['len'] / 10))\nprint(df['score'].round(1))\nprint(pd.DataFrame([[1, 2], [3, 4]], columns=['a', 'b'], index=['r1', 'r2']))",
    "stdout": "0    1\n1    0\n2    1\n3    0\n4    0\nName: label, dtype: int64\n  label  free  len  ratio\n0  spam     1   10    1.0\n1   ham     0   25    2.5\n2  spam     1   12    1.2\n3   ham     0   30    3.0\n4   ham     1    8    0.8\n0    88.5\n1    92.2\n2     NaN\n3    70.0\nName: score, dtype: float64\n    a  b\nr1  1  2\nr2  3  4\n"
  },
  "csv": {
    "src": "import pandas as pd\nimport numpy as np\ndf = pd.DataFrame({'name': ['Ann', 'Bob', 'Cy', 'Dee'], 'age': [23, 31, 19, 45], 'score': [88.5, 92.25, None, 70.0], 'passed': [True, True, False, True]})\nd2 = pd.DataFrame({'label': ['spam', 'ham', 'spam', 'ham', 'ham'], 'free': [1, 0, 1, 0, 1], 'len': [10, 25, 12, 30, 8]})\nimport io\ntext = 'x,y,cls\\n1,2.5,a\\n3,,b\\n5,7.5,a'\nt = pd.read_csv(io.StringIO(text))\nprint(t)\nprint(t.dtypes)\nprint(t.groupby('cls')['x'].sum())",
    "stdout": "   x    y cls\n0  1  2.5   a\n1  3  NaN   b\n2  5  7.5   a\nx        int64\ny      float64\ncls     object\ndtype: object\ncls\na    6\nb    3\nName: x, dtype: int64\n"
  },
  "newcol": {
    "src": "import pandas as pd\nimport numpy as np\ndf = pd.DataFrame({'name': ['Ann', 'Bob', 'Cy', 'Dee'], 'age': [23, 31, 19, 45], 'score': [88.5, 92.25, None, 70.0], 'passed': [True, True, False, True]})\nd2 = pd.DataFrame({'label': ['spam', 'ham', 'spam', 'ham', 'ham'], 'free': [1, 0, 1, 0, 1], 'len': [10, 25, 12, 30, 8]})\ndf['grade'] = np.where(df['age'] > 25, 'A', 'B') if False else ['B', 'A', 'B', 'A']\nprint(df)\ndf.loc[df['age'] > 30, 'senior'] = True\nprint(df[['name', 'senior']])",
    "stdout": "  name  age  score  passed grade\n0  Ann   23  88.50    True     B\n1  Bob   31  92.25    True     A\n2   Cy   19    NaN   False     B\n3  Dee   45  70.00    True     A\n  name senior\n0  Ann    NaN\n1  Bob   True\n2   Cy    NaN\n3  Dee   True\n"
  }
}

describe('pandas sandbox matches pandas', () => {
  for (const [name, c] of Object.entries(CASES)) {
    it(name, () => {
      const r = run(c.src)
      expect(r.error).toBeNull()
      expect(r.stdout).toBe(c.stdout)
    })
  }
})
