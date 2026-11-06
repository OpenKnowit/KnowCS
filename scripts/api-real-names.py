#!/usr/bin/env python3
"""The public names of the real libraries: what /playground/coverage/ counts against.

Run it twice, because no one Python here has everything, and it merges into scripts/api-real-names.json:

    python3 scripts/api-real-names.py                                # NumPy, pandas, matplotlib, sklearn, cv2, TF, Keras, CPython
    ~/repo/minimind/.venv/bin/python scripts/api-real-names.py torch # PyTorch

A name counts when it is in the object's __all__, or else public (no leading _). Submodules are left out, as each
has its own row, and so are modules a library merely imported.
"""
import importlib
import inspect
import json
import os
import pathlib
import platform
import sys
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter('ignore')
OUT = pathlib.Path(__file__).parent / 'api-real-names.json'


def names(obj):
    """an instance or class takes all its public attributes; a module takes __all__ if it declares one"""
    if inspect.ismodule(obj) and isinstance(getattr(obj, '__all__', None), (list, tuple)):
        found = [n for n in obj.__all__ if isinstance(n, str)]
    else:
        found = dir(obj)
    keep = []
    for n in found:
        if n.startswith('_'):
            continue
        try:
            if inspect.ismodule(getattr(obj, n)):
                continue
        except Exception:
            pass
        keep.append(n)
    return sorted(set(keep))


groups = {}


def add(lib, module, expr, obj, setup=''):
    groups[module] = {'lib': lib, 'expr': expr, 'setup': setup, 'names': names(obj)}


versions = {}
if sys.argv[1:] == ['torch']:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.utils.data as data
    import torch.distributions as dist
    versions['pytorch'] = torch.__version__
    S = 'import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nimport torch.utils.data\nimport torch.distributions\n'
    add('pytorch', 'torch', 'torch', torch, S)
    add('pytorch', 'Tensor', 'torch.ones(1)', torch.Tensor, S)
    add('pytorch', 'torch.nn', 'nn', nn, S)
    add('pytorch', 'nn.Module', 'nn.Linear(1, 1)', nn.Module, S)
    add('pytorch', 'nn.functional', 'F', F, S)
    add('pytorch', 'nn.init', 'nn.init', nn.init, S)
    add('pytorch', 'nn.utils', 'nn.utils', nn.utils, S)
    add('pytorch', 'torch.linalg', 'torch.linalg', torch.linalg, S)
    add('pytorch', 'torch.special', 'torch.special', torch.special, S)
    add('pytorch', 'torch.optim', 'torch.optim', torch.optim, S)
    add('pytorch', 'optim.lr_scheduler', 'torch.optim.lr_scheduler', torch.optim.lr_scheduler, S)
    add('pytorch', 'torch.autograd', 'torch.autograd', torch.autograd, S)
    add('pytorch', 'utils.data', 'torch.utils.data', data, S)
    add('pytorch', 'distributions', 'torch.distributions', dist, S)
else:
    import numpy as np
    versions['python'] = platform.python_version()
    versions['numpy'] = np.__version__
    add('numpy', 'np', 'np', np, 'import numpy as np\n')
    add('numpy', 'ndarray', 'np.arange(6.).reshape(2, 3)', np.ndarray, 'import numpy as np\n')
    add('numpy', 'np.random', 'np.random', np.random, 'import numpy as np\n')
    add('numpy', 'np.linalg', 'np.linalg', np.linalg, 'import numpy as np\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    versions['matplotlib'] = matplotlib.__version__
    S = 'import matplotlib.pyplot as plt\n'
    add('matplotlib', 'plt', 'plt', plt, S)
    add('matplotlib', 'Axes', 'plt.subplots()[1]', Axes, S)
    add('matplotlib', 'Figure', 'plt.subplots()[0]', Figure, S)

    import pandas as pd
    versions['pandas'] = pd.__version__
    S = "import pandas as pd\nD = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})\n"
    add('pandas', 'pd', 'pd', pd, S)
    add('pandas', 'DataFrame', 'D', pd.DataFrame, S)
    add('pandas', 'Series', "D['a']", pd.Series, S)
    add('pandas', 'Series.str', "pd.Series(['a']).str", pd.Series(['a']).str, S)
    add('pandas', 'GroupBy', "D.groupby('a')", pd.DataFrame({'a': [1]}).groupby('a'), S)

    import tensorflow as tf
    import keras
    versions['tensorflow'] = tf.__version__
    versions['keras'] = keras.__version__
    S = 'import numpy as np\nimport keras\nfrom keras import layers\n'
    add('keras', 'keras', 'keras', keras, S)
    add('keras', 'keras.layers', 'keras.layers', keras.layers, S)
    add('keras', 'keras.Model', 'keras.Sequential([layers.Input((1,)), layers.Dense(1)])', keras.Model, S)
    add('keras', 'keras.models', 'keras.models', keras.models, S)
    add('keras', 'keras.optimizers', 'keras.optimizers', keras.optimizers, S)
    add('keras', 'keras.losses', 'keras.losses', keras.losses, S)
    add('keras', 'keras.metrics', 'keras.metrics', keras.metrics, S)
    add('keras', 'keras.callbacks', 'keras.callbacks', keras.callbacks, S)
    add('keras', 'keras.activations', 'keras.activations', keras.activations, S)
    add('keras', 'keras.initializers', 'keras.initializers', keras.initializers, S)
    add('keras', 'keras.regularizers', 'keras.regularizers', keras.regularizers, S)
    add('keras', 'keras.utils', 'keras.utils', keras.utils, S)
    add('keras', 'keras.ops', 'keras.ops', keras.ops, S)
    S = 'import numpy as np\nimport tensorflow as tf\n'
    add('tensorflow', 'tf', 'tf', tf, S)
    add('tensorflow', 'tf.Tensor', 'tf.constant([1.])', tf.constant([1.]), S)
    add('tensorflow', 'tf.Variable', 'tf.Variable([1.])', tf.Variable([1.]), S)
    add('tensorflow', 'tf.math', 'tf.math', tf.math, S)
    add('tensorflow', 'tf.nn', 'tf.nn', tf.nn, S)
    add('tensorflow', 'tf.linalg', 'tf.linalg', tf.linalg, S)
    add('tensorflow', 'tf.random', 'tf.random', tf.random, S)
    add('tensorflow', 'tf.image', 'tf.image', tf.image, S)
    add('tensorflow', 'tf.strings', 'tf.strings', tf.strings, S)
    add('tensorflow', 'tf.config', 'tf.config', tf.config, S)
    add('tensorflow', 'tf.data.Dataset', 'tf.data.Dataset.range(2)', tf.data.Dataset.range(2), S)

    import sklearn
    versions['sklearn'] = sklearn.__version__
    for m in ['model_selection', 'preprocessing', 'pipeline', 'compose', 'neighbors', 'cluster', 'naive_bayes', 'linear_model', 'tree', 'decomposition', 'metrics', 'datasets']:
        mod = importlib.import_module(f'sklearn.{m}')
        add('sklearn', f'sklearn.{m}', 'M', mod, f'import sklearn.{m} as M\n')

    import cv2
    versions['cv2'] = cv2.__version__
    # OpenCV's constants would swamp its functions: they get a row of their own
    public = [n for n in dir(cv2) if not n.startswith('_') and not inspect.ismodule(getattr(cv2, n))]   # its __all__ is empty
    fn = [n for n in public if not n[0].isupper() and not n.startswith('CV_')]
    groups['cv2'] = {'lib': 'cv2', 'expr': 'cv2', 'setup': 'import cv2\n', 'names': fn}
    groups['cv2 constants'] = {'lib': 'cv2', 'expr': 'cv2', 'setup': 'import cv2\n', 'names': [n for n in public if n[0].isupper() and n not in fn]}

    import builtins
    S = ''
    groups['builtins'] = {'lib': 'python', 'expr': '__builtins__', 'setup': '', 'names': [n for n in names(builtins) if not n[0].isupper() or n.endswith(('Error', 'Exception', 'Warning', 'Exit', 'Interrupt', 'Iteration'))]}
    for label, value, py in [('str', 'abc', "'abc'"), ('list', [1], '[1]'), ('dict', {'a': 1}, "{'a': 1}"), ('set', {1}, '{1}'), ('tuple', (1,), '(1,)'), ('int', 1, '(1)'), ('float', 1.5, '1.5')]:
        add('python', label, py, value)
    for m in ['math', 'random', 're', 'json', 'collections', 'itertools', 'functools', 'operator', 'dataclasses', 'copy', 'os', 'os.path', 'sys', 'heapq', 'bisect', 'string', 'typing', 'time', 'abc', 'io', 'pickle', 'statistics', 'datetime', 'csv', 'argparse', 'pathlib', 'enum', 'contextlib', 'textwrap', 'types', 'tempfile', 'glob', 'shutil', 'fnmatch', 'inspect', 'fractions', 'uuid', 'secrets', 'platform', 'getpass']:
        add('python', m, 'M', importlib.import_module(m), f'import {m} as M\n')

old = json.loads(OUT.read_text()) if OUT.exists() else {'versions': {}, 'groups': {}}
old['versions'].update(versions)
old['groups'].update(groups)
OUT.write_text(json.dumps(old, indent=0, sort_keys=True) + '\n')
print(len(groups), 'groups,', sum(len(g['names']) for g in groups.values()), 'names', versions)
