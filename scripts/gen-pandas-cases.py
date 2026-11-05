#!/usr/bin/env python3
"""Runs each program in pandas-cases.txt with real pandas and freezes what it printed, or the error it ended on.

src/lib/pyPandasCases.test.ts then checks that the sandbox prints exactly the same. Output depends on the terminal
width, so this runs as a script with no terminal: 80 columns, as `python3 script.py > out.txt` would.
Run with no arguments after editing pandas-cases.txt.
"""
import contextlib
import io
import json
import os
import pathlib
import sys
import tempfile
import warnings

os.environ.pop('COLUMNS', None)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.simplefilter('ignore')
HEAD = 'import pandas as pd\nimport numpy as np\nimport io\n'
here = pathlib.Path(__file__).parent
cases = []
for block in (here / 'pandas-cases.txt').read_text().split('### ')[1:]:
    name, code = block.split('\n', 1)
    program = HEAD + code
    out = io.StringIO()
    error = None
    pd.reset_option('all')
    # an empty folder of its own: some programs write files
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            with contextlib.redirect_stdout(out):
                exec(compile(program, name, 'exec'), {'__name__': '__main__'})
        except BaseException as e:
            error = {'type': type(e).__name__, 'message': repr(e.args[0]) if isinstance(e, KeyError) and e.args else str(e)}
        os.chdir(here)
    cases.append({'name': name, 'code': program, 'out': out.getvalue(), 'error': error})
(here.parent / 'src/lib/pyPandasCases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=0) + '\n')
print(f'{len(cases)} programs on pandas {pd.__version__}, NumPy {np.__version__}, Python {sys.version.split()[0]}')
