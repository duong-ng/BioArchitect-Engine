import ast
import sys

files = [
    "step2_hf_folding_evaluator.py",
    "run_hf_lanmodulin_demo.py",
    "main_optimizer.py",
]

all_ok = True
for f in files:
    try:
        with open(f, "r", encoding="utf-8") as fh:
            source = fh.read()
        ast.parse(source)
        print(f"  [OK] {f}")
    except SyntaxError as e:
        print(f"  [FAIL] {f}: {e}")
        all_ok = False

if all_ok:
    print("\nAll files passed syntax check!")
else:
    print("\nSome files have syntax errors!")
    sys.exit(1)
