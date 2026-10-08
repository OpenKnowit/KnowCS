# Runs the editor's code for the KnowCS playground inside Pyodide (see worker.ts).
# Each run gets a fresh namespace; print/stderr are captured; the value of a last bare expression is shown like
# a notebook cell; matplotlib figures (plt.show() or left open) come back as PNG data URLs.
import ast
import base64
import io
import json
import linecache
import os
import sys
import traceback
import warnings

os.environ.setdefault("MPLBACKEND", "AGG")
# Pyodide's own deprecation noise from inside sklearn's threadpoolctl, not something the student's code did
warnings.filterwarnings("ignore", message=r"JsProxy\.as_object_map", category=RuntimeWarning)

_FILE = "<editor>"
_MAX_TEXT = 100_000
_MAX_FIGURES = 8


class _Capped(io.StringIO):
    """a stdout that stops growing after _MAX_TEXT characters (a print in a long loop must not eat the tab)"""

    def __init__(self):
        super().__init__()
        self.cut = False

    def write(self, s):
        room = _MAX_TEXT - self.tell()
        if room <= 0:
            self.cut = True
            return len(s)
        if len(s) > room:
            self.cut = True
            s = s[:room]
        return super().write(s)

    def text(self):
        return self.getvalue() + ("\n…" if self.cut else "")


_figures = []


def _grab_figures():
    """save every open figure as a PNG and close it"""
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is None:
        return
    for num in plt.get_fignums():
        fig = plt.figure(num)
        if len(_figures) < _MAX_FIGURES:
            buf = io.BytesIO()
            with warnings.catch_warnings():  # the harness's own saving must not add warnings to the code's stderr
                warnings.simplefilter("ignore")
                fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
            _figures.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii"))
        plt.close(fig)


def _prepare_matplotlib():
    import matplotlib

    matplotlib.use("agg")
    import matplotlib.pyplot as plt

    plt.close("all")
    plt.show = lambda *args, **kwargs: _grab_figures()


def _user_traceback(e):
    """the traceback from the first frame in the editor's code (the harness's own frames are hidden)"""
    tb = e.__traceback__
    while tb is not None and tb.tb_frame.f_code.co_filename != _FILE:
        tb = tb.tb_next
    line = None
    if isinstance(e, SyntaxError) and e.filename == _FILE:
        line = e.lineno
    else:
        for frame, lineno in traceback.walk_tb(tb):
            if frame.f_code.co_filename == _FILE:
                line = lineno
    text = "".join(traceback.format_exception(type(e), e, tb)).rstrip()
    return {"type": type(e).__name__, "message": str(e), "line": line, "trace": text}


def _is_artist(x):
    if isinstance(x, (list, tuple)) and x:
        x = x[0]
    return type(x).__module__.startswith("matplotlib.")


def knowcs_run(src, uses_matplotlib):
    del _figures[:]
    out, err = _Capped(), _Capped()
    value = None
    error = None
    linecache.cache[_FILE] = (len(src), None, src.splitlines(True), _FILE)
    old = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        if uses_matplotlib:
            _prepare_matplotlib()
        tree = ast.parse(src, _FILE, "exec")
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)
        scope = {"__name__": "__main__", "__builtins__": __builtins__}
        exec(compile(tree, _FILE, "exec"), scope)
        if last is not None:
            result = eval(compile(last, _FILE, "eval"), scope)
            # a plot call's return value (Line2D lists, BarContainer …) is noise beside the figure itself
            if result is not None and not _is_artist(result):
                value = repr(result)
                if len(value) > _MAX_TEXT:
                    value = value[:_MAX_TEXT] + "\n…"
        _grab_figures()
    except BaseException as e:  # SystemExit and KeyboardInterrupt are the code's to raise too
        error = _user_traceback(e)
        try:
            _grab_figures()
        except Exception:
            pass
    finally:
        sys.stdout, sys.stderr = old
    return json.dumps({"stdout": out.text(), "stderr": err.text(), "value": value, "figures": list(_figures), "error": error})


def knowcs_versions():
    """the versions of the course libraries that are loaded"""
    found = {"python": sys.version.split()[0]}
    for name in ("numpy", "pandas", "matplotlib", "sklearn", "scipy", "cv2", "seaborn"):
        mod = sys.modules.get(name)
        if mod is not None:
            found[name] = getattr(mod, "__version__", "")
    return json.dumps(found)


def knowcs_missing(src):
    """the third-party modules src imports that are not imported yet (what the worker must download first)"""
    try:
        from pyodide.code import find_imports

        names = find_imports(src)
    except Exception:  # a syntax error: let the run report it
        return "[]"
    import importlib.util

    def installed(name):
        # Pyodide's import hook raises (not returns None) for a package it ships but has not loaded yet
        try:
            return importlib.util.find_spec(name) is not None
        except ImportError:
            return False

    tops = dict.fromkeys(n.split(".")[0] for n in names)  # sklearn.cluster → sklearn, in order, once
    return json.dumps([n for n in tops if n not in sys.stdlib_module_names and not installed(n)])
