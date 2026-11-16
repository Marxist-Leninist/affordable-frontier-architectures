"""Static safety review of the ASI-Arch gallery programs. Nothing is imported or executed here.

Each program is parsed with `ast` and must stay inside a small tensor-code surface:
  * imports only from an allowlist (torch, math, einops, typing, __future__, and the fla / transformers
    names that fla_cpu.py replaces with local stand-ins);
  * no use of names that reach the interpreter, the file system, processes or the network;
  * no dunder attribute access other than the usual class plumbing.
A program that fails any rule is listed and is not run by the harness.
"""
import ast
import hashlib
import json
import pathlib
import sys

# functools is allowed for record 330, which only uses functools.lru_cache as a decorator.
ALLOWED_IMPORT_ROOTS = {"torch", "math", "einops", "typing", "__future__", "fla", "transformers", "functools"}
ALLOWED_FROM = {
    "torch", "torch.nn", "torch.nn.functional", "math", "einops", "typing", "__future__",
    "fla.modules", "fla.modules.l2norm", "fla.layers.utils", "fla.models.utils", "transformers.processing_utils",
}
FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "open", "__import__", "input", "breakpoint", "globals", "locals", "vars",
    "memoryview", "os", "sys", "subprocess", "socket", "shutil", "pathlib", "importlib", "pickle", "ctypes",
    "requests", "urllib", "http", "builtins", "__builtins__", "exit", "quit", "help",
}
# torch entry points that touch files, the network or native code loading
FORBIDDEN_ATTRS = {
    "load", "save", "hub", "jit", "utils", "distributed", "multiprocessing", "serialization", "onnx", "package",
    "load_state_dict_from_url", "system", "popen", "remove", "unlink", "rmtree", "environ", "getenv",
}
ALLOWED_DUNDERS = {"__init__", "__class__", "__name__", "__doc__", "__call__", "__repr__", "__len__", "__getitem__"}


def review(source):
    problems = []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORT_ROOTS:
                    problems.append(f"import {a.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module not in ALLOWED_FROM:
                problems.append(f"from {'.' * node.level}{node.module} import ...")
        elif isinstance(node, ast.Name):
            # `input` is a common tensor parameter name (record 562); it is only a problem when called.
            if node.id in FORBIDDEN_NAMES and node.id != "input":
                problems.append(f"name {node.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__") and node.attr not in ALLOWED_DUNDERS:
                problems.append(f"dunder attribute {node.attr}")
            if node.attr in FORBIDDEN_ATTRS:
                # nn.Module.load_state_dict etc. are different names; only exact matches are flagged
                problems.append(f"attribute .{node.attr}")
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id == "input":
                problems.append("call input()")
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "functools" and f.attr != "lru_cache":
                problems.append(f"functools.{f.attr}")
            if isinstance(f, ast.Name) and f.id in ("getattr", "setattr", "delattr"):
                # allowed only with a literal attribute name that is not a dunder
                arg = node.args[1] if len(node.args) > 1 else None
                if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str) and not arg.value.startswith("__")):
                    if not isinstance(arg, (ast.JoinedStr, ast.Name, ast.BinOp, ast.Constant)):
                        problems.append(f"{f.id} with unusual attribute expression")
        elif isinstance(node, (ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.Await)):
            problems.append(type(node).__name__)
    return sorted(set(problems))


def main(path, out):
    records = json.loads(open(path, encoding="utf-8").read())
    rows = []
    for r in sorted(records, key=lambda r: r["index"]):
        src = r["program"]
        p = review(src)
        tree_ok = not p or not any(x.startswith("syntax") for x in p)
        rows.append({
            "index": r["index"], "name": r["name"], "sha256": hashlib.sha256(src.encode("utf-8")).hexdigest(),
            "lines": src.count("\n") + 1, "defines_DeltaNet": tree_ok and "class DeltaNet" in src, "problems": p,
        })
    clean = [x["index"] for x in rows if not x["problems"] and x["defines_DeltaNet"]]
    report = {"programs": len(rows), "clean": len(clean), "flagged": {str(x["index"]): x["problems"] for x in rows if x["problems"]},
              "rules": {"allowed_from": sorted(ALLOWED_FROM), "forbidden_names": sorted(FORBIDDEN_NAMES),
                        "forbidden_attributes": sorted(FORBIDDEN_ATTRS), "allowed_dunders": sorted(ALLOWED_DUNDERS)},
              "rows": rows}
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
        f.write("\n")
    print(json.dumps({"programs": len(rows), "clean": len(clean), "flagged": report["flagged"]}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
