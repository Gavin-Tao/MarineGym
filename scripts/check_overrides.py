#!/usr/bin/env python
"""预检：命令行里同一个 hydra 键出现多次且取值不同就报错。
两次事故(speed 被反压、seed 被反压)都属于这一类，靠约定防不住，靠这个防。"""
import sys, collections
seen = collections.OrderedDict()
for tok in sys.argv[1:]:
    if "=" not in tok or tok.startswith("-"): continue
    k, v = tok.split("=", 1)
    seen.setdefault(k, []).append(v)
bad = {k: v for k, v in seen.items() if len(set(v)) > 1}
if bad:
    print("!!! 覆盖冲突（后者生效，很可能不是你要的）:", file=sys.stderr)
    for k, v in bad.items():
        print(f"    {k} = {' -> '.join(v)}   实际生效: {v[-1]}", file=sys.stderr)
    sys.exit(1)
dup = {k: v for k, v in seen.items() if len(v) > 1}
if dup:
    print(f"(提示: {len(dup)} 个键重复但取值一致，无害)", file=sys.stderr)
sys.exit(0)
