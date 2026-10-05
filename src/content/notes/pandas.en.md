# pandas API Landscape

> Measured on pandas 2.3.3 (2026-06)
> Bottom line: **2,500 public APIs ≈ 3 core classes × a 7-step pipeline + 4 accessors + heavy reuse of names and naming patterns**. Fewer than 50 actually need to be memorised.

## 1. API Counts

| Level | Count | Notes |
|---|---|---|
| All public APIs (deduplicated) | **~2,499** | Top level, class members, accessors, submodules |
| `pd.*` top level | 119 | 61 functions + 41 classes |
| `pd.DataFrame` | 209 | 192 callable methods |
| `pd.Series` | 210 | 187 callable methods |
| `pd.Index` | 101 | |
| `Series.str.*` | 56 | String accessor |
| `Series.dt.*` | 42 | Datetime accessor |
| `pd.api.types` / `pd.errors` / `pd.tseries.offsets` | 46 / 43 / 43 | Long tail, look up when needed |

## 2. Mental Map: Four Layers

### 1. Only 3 core objects

```
Series      one labelled column of data (1-D) = a NumPy array with labels
DataFrame   a table (2-D) = a dict of Series sharing one Index
Index       the label axis for rows / columns
```

### 2. DataFrame and Series share 86% of their methods

Both inherit from the same base class `NDFrame`: **179 of the 209 DataFrame members have the same name and meaning on Series**, so learning them once covers both.

- **DataFrame only (30)** = concepts that exist only for 2-D tables: `merge` `join` `pivot` `melt` `stack` `set_index` `assign` `query` `columns` `to_html` `to_parquet` …
- **Series only (31)** = 1-D array semantics: the `.str` `.dt` `.cat` accessors, `unique` `tolist` `argmax` `name` `dtype` (singular), `to_frame` …

⚠️ **Trap: methods with the same name but different behaviour**:

| Method | On a Series | On a DataFrame |
|---|---|---|
| `apply(f)` | f receives each **element** | f receives each **column / row** (a whole Series) |
| `map(f)` | element-wise | element-wise, only from 2.1 (formerly `applymap`) |
| `rename(x)` | changes the name / index | changes row / column labels |
| `drop(x)` | drops index entries | drops rows by default; use `axis=1` or `columns=` for columns |
| statistics methods | no notion of axis | take `axis=0/1` |

**`axis` is the key to understanding every difference.**

### 3. Four namespace accessors (~110 APIs, barely need learning)

```python
s.str.*    # 56, vectorised versions of Python str methods (contains/split/lower)
s.dt.*     # 42, datetime attributes carried over (year/month/dayofweek)
s.cat.*    # 11, categorical operations
df.plot.*  # a thin wrapper around matplotlib
```

### 4. The family of deferred objects

They don't return a result directly; they return an intermediate object waiting for aggregation, and the follow-up methods are the same as DataFrame's statistics methods:

```python
df.groupby('a')    → GroupBy     grouping
df.rolling(7)      → Rolling     sliding window
df.resample('ME')  → Resampler   time resampling
df.expanding()     → Expanding   cumulative window
```

> Historical footnote: the name pandas comes from **pan**el **da**ta. The 3-D `Panel` structure was deprecated in 0.20 and **removed in 0.25 (2019)**. Replacements: grouped tabular data → a MultiIndex DataFrame; true N-dimensional arrays → xarray.

---

## 3. The Core: the Data Lifecycle Pipeline

```
read → inspect → select → clean → reshape → compute → output
```

When stuck, first ask "which pipeline step am I in?", then look for a method in that group.

### ① Read: `pd.read_*` (20)

```
Text:        read_csv  read_table  read_fwf  read_clipboard
Spreadsheet: read_excel  read_html
Structured:  read_json  read_xml
Binary:      read_parquet  read_feather  read_orc  read_pickle  read_hdf
Database:    read_sql  read_sql_query  read_sql_table  read_gbq
Stats tools: read_sas  read_spss  read_stata
```

90% of the time you only use `read_csv`. Key parameters:

```python
pd.read_csv('f.csv',
    sep=',',              # delimiter
    header=0,             # row holding the column names; None = no header
    index_col='id',       # column to use as the index
    usecols=['a','b'],    # read only these columns (faster for big files)
    dtype={'code': str},  # force a type: stops "001" being read as 1
    parse_dates=['date'], # parse dates automatically
    nrows=1000,           # read only the first N rows (probe a big file)
    chunksize=50000,      # iterate in chunks (when memory is short)
    na_values=['-','NA'], # custom missing-value markers
)
```

Pitfalls and experience:
- Postcodes / stock codes lose leading zeros → `dtype=str`
- `read_excel` needs `openpyxl`, `read_parquet` needs `pyarrow`
- For intermediate data, prefer **parquet** (fast, keeps types, small files) over csv

### ② Inspect: the "five moves" after loading data (read-only, type freely)

```python
df.head(10)        # first rows
df.info()          # row count, dtypes, non-null counts, memory ← the most important
df.describe()      # numeric summary; include='object' for text columns
df.shape           # (rows, columns)
df['col'].value_counts(dropna=False)   # value distribution ← the tool for spotting dirty data
```

Also: `df.dtypes`, `df.nunique()`, `df.sample(5)` (exposes problems better than head), `df.isna().sum()`.

### ③ Select: two systems

**By label vs by position:**

```python
df.loc['2026-01-01', 'price']   # by label
df.iloc[0, 2]                   # by position (pure integers)
df.at[...] / df.iat[...]        # fast single-value versions
```

**Boolean filtering:**

```python
df[df['price'] > 100]
df[(df['a'] > 1) & (df['b'] < 5)]            # use & | ~, and parenthesise every condition!
df[df['city'].isin(['Beijing', 'Shanghai'])]
df.query('price > 100 and city == "Beijing"')   # easier to read for long conditions
df.filter(like='2026', axis=1)               # select columns by fuzzy name match
df.select_dtypes(include='number')           # select columns by dtype
```

Pitfalls:
- `df[0:3]` slices rows while `df['a']` selects a column: the same `[]` behaves differently. **Always write loc/iloc explicitly.**
- `SettingWithCopyWarning`: write `df.loc[cond, 'b'] = 0`, not `df[cond]['b'] = 0`
- Using `and`/`or` in conditions raises an error; you must use `&`/`|` with parentheses

### ④ Clean: 60% of real-world work

```python
# the missing-value trio
df.isna().sum()                     # first see what is missing
df.dropna(subset=['key_col'])       # drop rows missing a key column
df.fillna({'age': df['age'].median(), 'city': 'unknown'})
df['x'].ffill()                     # forward fill (common for time series)

# duplicates
df.duplicated(subset=['id']).sum()
df.drop_duplicates(subset=['id'], keep='last')

# fixing types (the most common source of dirty data)
df['price'] = pd.to_numeric(df['price'], errors='coerce')   # unconvertible values become NaN
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['grade'] = df['grade'].astype('category')                # saves memory for low-cardinality columns

# replacing values and renaming
df['sex'].replace({'M': 'male', 'F': 'female'})
df.rename(columns={'old': 'new'})
df.columns = df.columns.str.strip().str.lower()             # clean all column names at once

# outliers
df['price'].clip(lower=0, upper=df['price'].quantile(0.99))
```

**Rule of thumb**: first use `to_numeric` / `to_datetime` with `errors='coerce'` to turn every dirty value into NaN, then handle them with the missing-value trio. Two steps.

### ⑤ Reshape: 4 pairs of opposite operations

**Join side by side vs stack vertically:**

```python
pd.merge(orders, users, on='user_id', how='left')   # SQL JOIN; how: inner/left/right/outer
pd.concat([df1, df2])                               # stack vertically; axis=1 joins side by side
```
> Pitfall: more rows after a merge = duplicate keys in the right table. Check `users['user_id'].is_unique` first.

**Long ↔ wide (inverses):**

```python
df.pivot_table(index='date', columns='city', values='temp', aggfunc='mean')  # long → wide
df.melt(id_vars='date', var_name='city', value_name='temp')                  # wide → long
```
> `pivot` errors out when combinations aren't unique, so in practice just use `pivot_table`.

**Column ↔ index (inverses):**

```python
df.set_index('date') / df.reset_index()    # reset_index almost always follows a groupby
df.stack() / df.unstack()                  # column labels ↔ row index (MultiIndex)
```

**Grouping (split-apply-combine):**

```python
df.groupby('city')['price'].mean()
df.groupby(['city', 'year']).agg(
    avg_price=('price', 'mean'),    # named aggregation
    n=('order_id', 'count'),
)
```
> `pivot_table` is essentially a shortcut for `groupby + unstack`.

**Reordering:**

```python
df.sort_values(['city', 'price'], ascending=[True, False])
df.nlargest(10, 'price')    # faster than sort followed by head
```

### ⑥ Compute: the "what to compute × over what range" matrix

| Range \ what | Built-in statistics | Custom function |
|---|---|---|
| Whole column (reduces) | `df['x'].sum()` | `df['x'].agg(f)` |
| Per row / element (same shape) | `cumsum` `diff` `pct_change` `rank` | `map` / `apply` |
| Per group | `groupby(...).sum()` | `.agg(f)` / `.transform(f)` |
| Sliding window | `rolling(7).mean()` | `rolling(7).apply(f)` |
| Time resampling | `resample('ME').sum()` | `resample('ME').agg(f)` |

Three easily confused ways to apply functions:

```python
df['x'].map(f)          # element-wise, 1 in 1 out
df.apply(f)             # receives one whole column Series at a time
df.groupby('g')['x'].transform('mean')   # aggregated values broadcast back to the original shape
```

```python
# classic transform: each row's amount as a share of its city's total
df['ratio'] = df['amt'] / df.groupby('city')['amt'].transform('sum')
```

**Performance rule**: if a built-in (vectorised) method exists, don't `apply`; the difference is 10–100×. `apply` is an escape hatch, not the default.

### ⑦ Output: `to_*` (~20, mirroring read_*)

```
File mirrors: to_csv  to_excel  to_json  to_parquet  to_pickle  to_sql  to_html ...
To objects:   to_dict  to_numpy  to_list(Series)  to_frame(Series)
Other:        to_clipboard (paste straight into Excel, great for debugging)  to_string  to_markdown
```

```python
df.to_csv('out.csv', index=False)      # index=False is what you want most of the time!
df.to_parquet('out.parquet')           # first choice for intermediate pipeline results
df.to_dict(orient='records')           # [{row 1}, {row 2}, ...] for APIs / JSON
```

> Pitfall: forget `index=False` and you get an extra `Unnamed: 0` column the next time you read it.

---

## 4. Putting It Together: a Complete Pipeline in Five Lines (chained style)

```python
(pd.read_csv('orders.csv', parse_dates=['date'])        # ① read
   .query('amount > 0')                                  # ③ select
   .assign(amount=lambda d: d['amount'].fillna(0))       # ④ clean
   .groupby([pd.Grouper(key='date', freq='ME'), 'city']) # ⑤ reshape
   .agg(total=('amount', 'sum'))                         # ⑥ compute
   .reset_index()
   .to_parquet('monthly.parquet'))                       # ⑦ output
```

Method chaining is idiomatic pandas, which also explains why almost all methods **return a new object instead of modifying in place**.

## 5. Learning Strategy

1. **Memorise 30 core APIs first** (the representative ones in the pipeline sections), which cover 90% of daily work
2. **Infer from naming patterns**: `read_*` / `to_*` / `is*` / `sort_*` / `drop*` tell you what they do by name
3. **When stuck, locate the pipeline step first**, then look for a method in that group
4. Practice: take a real csv and force yourself through the whole pipeline using only the APIs above
5. The official [10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html) is organised along exactly this pipeline

## Appendix: Pitfall Checklist

- [ ] `dtype=str` to keep leading zeros
- [ ] Boolean filters use `&` `|` `~` + parentheses, never `and`/`or`
- [ ] Assign with `df.loc[cond, col] = val` to avoid SettingWithCopyWarning
- [ ] Check key uniqueness before merging: `s.is_unique`
- [ ] `to_csv(index=False)`, otherwise you get an extra `Unnamed: 0`
- [ ] Don't `apply` when you can vectorise
- [ ] `errors='coerce'` turns dirty values into NaN first, then handle them together
- [ ] pandas ≥ 0.25 has no `pd.Panel`; use a MultiIndex DataFrame or xarray
