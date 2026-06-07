# pandas API 全景地图

> 基于 pandas 2.3.3 实测整理（2026-06）
> 核心结论：**2,500 个公开 API ≈ 3 个核心类 × 7 步流水线 + 4 个访问器 + 大量同名复用和命名规律**，真正需要主动记忆的不到 50 个。

## 一、API 数量统计

| 层级 | 数量 | 说明 |
|---|---|---|
| 全部公开 API（去重） | **~2,499** | 含顶层、类成员、访问器、子模块 |
| `pd.*` 顶层 | 119 | 函数 61 个 + 类 41 个 |
| `pd.DataFrame` | 209 | 可调用方法 192 个 |
| `pd.Series` | 210 | 可调用方法 187 个 |
| `pd.Index` | 101 | |
| `Series.str.*` | 56 | 字符串访问器 |
| `Series.dt.*` | 42 | 日期时间访问器 |
| `pd.api.types` / `pd.errors` / `pd.tseries.offsets` | 46 / 43 / 43 | 长尾，按需查 |

## 二、心智地图：四层结构

### 1. 只有 3 个核心对象

```
Series      一列带标签的数据（一维）= 带标签的 NumPy 数组
DataFrame   一张表（二维）= 共享同一个 Index 的 Series 字典
Index       行/列的标签轴
```

### 2. DataFrame 与 Series 共享 86% 的方法

两者继承自同一基类 `NDFrame`，**209 个 DataFrame 成员中 179 个与 Series 同名同义**——学一遍通用两边。

- **DataFrame 独有（30 个）**= 只有二维表才有的概念：`merge` `join` `pivot` `melt` `stack` `set_index` `assign` `query` `columns` `to_html` `to_parquet` …
- **Series 独有（31 个）**= 一维数组语义：`.str` `.dt` `.cat` 访问器、`unique` `tolist` `argmax` `name` `dtype`（单数）、`to_frame` …

⚠️ **同名但行为有差异的陷阱方法**：

| 方法 | Series 上 | DataFrame 上 |
|---|---|---|
| `apply(f)` | f 收到每个**元素** | f 收到每一**列/行**（整个 Series） |
| `map(f)` | 逐元素 | 2.1+ 才有，逐元素（旧版叫 `applymap`） |
| `rename(x)` | 改 name/索引 | 改行/列标签 |
| `drop(x)` | 删索引项 | 默认删行，删列要 `axis=1` 或 `columns=` |
| 统计方法 | 无轴概念 | 有 `axis=0/1` |

**`axis` 是理解一切差异的钥匙。**

### 3. 四个命名空间访问器（~110 个 API，基本不用学）

```python
s.str.*    # 56 个，Python str 方法的向量化版（contains/split/lower）
s.dt.*     # 42 个，datetime 属性搬过来（year/month/dayofweek）
s.cat.*    # 11 个，分类操作
df.plot.*  # matplotlib 薄封装
```

### 4. 延迟对象家族

不直接返回结果，返回中间对象等待聚合，后续方法与 DataFrame 统计方法是同一套：

```python
df.groupby('a')    → GroupBy     分组
df.rolling(7)      → Rolling     滑动窗口
df.resample('ME')  → Resampler   时间重采样
df.expanding()     → Expanding   累积窗口
```

> 历史注脚：pandas 名字来自 **pan**el **da**ta。三维结构 `Panel` 于 0.20 弃用、**0.25 (2019) 移除**。替代：分组表格数据 → MultiIndex DataFrame；真正的 N 维数组 → xarray。

---

## 三、核心：数据生命周期流水线

```
读入 → 查看 → 选取 → 清洗 → 变形 → 计算 → 输出
```

遇事先想「我在流水线哪一步」，再去那一类里找方法。

### ① 读入 `pd.read_*`（20 个）

```
文本类:   read_csv  read_table  read_fwf  read_clipboard
表格类:   read_excel  read_html
结构化:   read_json  read_xml
二进制:   read_parquet  read_feather  read_orc  read_pickle  read_hdf
数据库:   read_sql  read_sql_query  read_sql_table  read_gbq
统计软件: read_sas  read_spss  read_stata
```

90% 的时间只用 `read_csv`，关键参数：

```python
pd.read_csv('f.csv',
    sep=',',              # 分隔符
    header=0,             # 列名所在行；None = 无列名
    index_col='id',       # 哪列做索引
    usecols=['a','b'],    # 只读这几列（大文件提速）
    dtype={'code': str},  # 强制类型 —— 防止 "001" 被读成 1
    parse_dates=['date'], # 自动解析日期
    nrows=1000,           # 只读前 N 行（探查大文件）
    chunksize=50000,      # 分块迭代（内存不够时）
    na_values=['-','NA'], # 自定义缺失值标记
)
```

坑与经验：
- 邮编/股票代码丢前导零 → `dtype=str`
- `read_excel` 需 `openpyxl`，`read_parquet` 需 `pyarrow`
- 中间数据交换首选 **parquet**（快、保类型、文件小），别用 csv

### ② 查看 —— 拿到数据后的「五连招」（只读，随便敲）

```python
df.head(10)        # 看前几行
df.info()          # 行数、类型、非空数、内存 ← 最重要
df.describe()      # 数值摘要；include='object' 看文本列
df.shape           # (行数, 列数)
df['col'].value_counts(dropna=False)   # 取值分布 ← 探查脏数据神器
```

补充：`df.dtypes`、`df.nunique()`、`df.sample(5)`（比 head 更暴露问题）、`df.isna().sum()`。

### ③ 选取 —— 两套体系

**按标签 vs 按位置：**

```python
df.loc['2026-01-01', 'price']   # 标签
df.iloc[0, 2]                   # 位置（纯整数）
df.at[...] / df.iat[...]        # 单值快速版
```

**布尔过滤：**

```python
df[df['price'] > 100]
df[(df['a'] > 1) & (df['b'] < 5)]            # 用 & | ~，每个条件必须加括号！
df[df['city'].isin(['北京', '上海'])]
df.query('price > 100 and city == "北京"')   # 长条件更易读
df.filter(like='2026', axis=1)               # 按列名模糊选列
df.select_dtypes(include='number')           # 按类型选列
```

坑：
- `df[0:3]` 切行、`df['a']` 选列 —— 同一个 `[]` 行为不同，**推荐永远显式写 loc/iloc**
- `SettingWithCopyWarning`：要写 `df.loc[cond, 'b'] = 0`，不要 `df[cond]['b'] = 0`
- 条件用 `and`/`or` 会报错，必须 `&`/`|` 加括号

### ④ 清洗 —— 真实工作 60% 的时间

```python
# 缺失值三连
df.isna().sum()                     # 先看哪里缺
df.dropna(subset=['key_col'])       # 关键列缺失的行丢掉
df.fillna({'age': df['age'].median(), 'city': '未知'})
df['x'].ffill()                     # 前值填充（时间序列常用）

# 重复值
df.duplicated(subset=['id']).sum()
df.drop_duplicates(subset=['id'], keep='last')

# 类型修正（脏数据最常见病灶）
df['price'] = pd.to_numeric(df['price'], errors='coerce')   # 转不了的变 NaN
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['grade'] = df['grade'].astype('category')                # 低基数列省内存

# 值替换与改名
df['sex'].replace({'M': '男', 'F': '女'})
df.rename(columns={'old': 'new'})
df.columns = df.columns.str.strip().str.lower()             # 批量清洗列名

# 异常值
df['price'].clip(lower=0, upper=df['price'].quantile(0.99))
```

**心法**：先用 `to_numeric`/`to_datetime` + `errors='coerce'` 把脏值统一变 NaN，再用缺失值三连处理。两步走。

### ⑤ 变形 —— 4 组对立操作

**横向拼 vs 纵向拼：**

```python
pd.merge(orders, users, on='user_id', how='left')   # SQL JOIN；how: inner/left/right/outer
pd.concat([df1, df2])                               # 纵向摞；axis=1 可横拼
```
> 坑：merge 后行数变多 = 右表 key 有重复，先 `users['user_id'].is_unique` 验证。

**长表 ↔ 宽表（互逆）：**

```python
df.pivot_table(index='date', columns='city', values='temp', aggfunc='mean')  # 长→宽
df.melt(id_vars='date', var_name='city', value_name='temp')                  # 宽→长
```
> `pivot` 要求组合唯一会报错，日常直接用 `pivot_table`。

**列 ↔ 索引（互逆）：**

```python
df.set_index('date') / df.reset_index()    # groupby 之后几乎必接 reset_index
df.stack() / df.unstack()                  # 列标签 ↔ 行索引（MultiIndex）
```

**分组（split-apply-combine）：**

```python
df.groupby('city')['price'].mean()
df.groupby(['city', 'year']).agg(
    avg_price=('price', 'mean'),    # 命名聚合
    n=('order_id', 'count'),
)
```
> `pivot_table` 本质 = `groupby + unstack` 的快捷方式。

**重排：**

```python
df.sort_values(['city', 'price'], ascending=[True, False])
df.nlargest(10, 'price')    # 比 sort 后 head 更快
```

### ⑥ 计算 —— 「算什么 × 在什么范围算」矩阵

| 范围 \ 算什么 | 内置统计 | 自定义函数 |
|---|---|---|
| 整列（降维） | `df['x'].sum()` | `df['x'].agg(f)` |
| 逐行/逐元素（形状不变） | `cumsum` `diff` `pct_change` `rank` | `map` / `apply` |
| 每个分组 | `groupby(...).sum()` | `.agg(f)` / `.transform(f)` |
| 滑动窗口 | `rolling(7).mean()` | `rolling(7).apply(f)` |
| 时间重采样 | `resample('ME').sum()` | `resample('ME').agg(f)` |

三个易混的函数应用：

```python
df['x'].map(f)          # 逐元素，1 进 1 出
df.apply(f)             # 每次收到一整列 Series
df.groupby('g')['x'].transform('mean')   # 聚合值广播回原表形状
```

```python
# transform 经典用法：每行金额占所在城市总额的比例
df['ratio'] = df['amt'] / df.groupby('city')['amt'].transform('sum')
```

**性能心法**：能用内置方法（向量化）就别 `apply`，差 10~100 倍。`apply` 是逃生舱，不是默认选择。

### ⑦ 输出 `to_*`（~20 个，与 read_* 镜像）

```
文件镜像: to_csv  to_excel  to_json  to_parquet  to_pickle  to_sql  to_html ...
转对象:   to_dict  to_numpy  to_list(Series)  to_frame(Series)
其他:     to_clipboard（直接粘进 Excel，调试神器） to_string  to_markdown
```

```python
df.to_csv('out.csv', index=False)      # index=False 最常用！
df.to_parquet('out.parquet')           # 流水线中间结果首选
df.to_dict(orient='records')           # [{行1}, {行2}, ...] 给 API/JSON 用
```

> 坑：忘了 `index=False`，下次读回来多出 `Unnamed: 0` 列。

---

## 四、串起来：五行完整流水线（链式风格）

```python
(pd.read_csv('orders.csv', parse_dates=['date'])        # ① 读入
   .query('amount > 0')                                  # ③ 选取
   .assign(amount=lambda d: d['amount'].fillna(0))       # ④ 清洗
   .groupby([pd.Grouper(key='date', freq='ME'), 'city']) # ⑤ 变形
   .agg(total=('amount', 'sum'))                         # ⑥ 计算
   .reset_index()
   .to_parquet('monthly.parquet'))                       # ⑦ 输出
```

链式写法是 pandas 惯用风格——这也解释了为什么绝大多数方法**返回新对象而不是原地修改**。

## 五、学习策略

1. **先背 30 个核心 API**（流水线表格中的加粗代表），覆盖 90% 日常
2. **靠命名规律推断**：`read_*` / `to_*` / `is*` / `sort_*` / `drop*` 看名字即知用途
3. **遇事先定位流水线阶段**，再去那一类里找方法
4. 练习：拿真实 csv，强迫自己每阶段只用上面的 API 走完一遍流水线
5. 官方 [10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html) 正是按此流水线组织

## 附：易踩坑清单速查

- [ ] `dtype=str` 防前导零丢失
- [ ] 布尔过滤用 `&` `|` `~` + 括号，不能用 `and`/`or`
- [ ] 赋值用 `df.loc[cond, col] = val`，避免 SettingWithCopyWarning
- [ ] merge 前验证 key 唯一性：`s.is_unique`
- [ ] `to_csv(index=False)`，否则多出 `Unnamed: 0`
- [ ] 能向量化就不 `apply`
- [ ] `errors='coerce'` 先把脏值变 NaN 再统一处理
- [ ] pandas ≥ 0.25 没有 `pd.Panel`，用 MultiIndex DataFrame 或 xarray
