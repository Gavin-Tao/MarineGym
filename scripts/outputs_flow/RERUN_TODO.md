# 待补跑

## ppo / calm / s0（K64N10）

结果行有效（`EVAL-ONLY RESULTS` 已产出），但**缺 `.episodes.csv`** ——
它在 `+ep_dump` 接进 `exp_eval.sh` 之前就启动了，参数在启动时就已固定。
显著性检验需要它。矩阵跑完后补：

```bash
rm -f outputs_flow/eval_K64N10/ppo__calm__s0.log
CKPT=... MPPI_K=64 MPPI_N=10 NUM_ENVS=64 EP=200 MB=20 bash exp_eval.sh ppo calm 0
```

---

## 教训：**不要编辑正在执行的 shell 脚本**

bash 是按**字节偏移**增量读取脚本的。在它执行中途修改文件，会让它从错位的位置
继续读，报出 `unexpected EOF while looking for matching '"'` 之类与真实代码无关的
语法错误。本次就是这样"弄挂"了 ppo/calm 那一格的收尾（评测本身已完成）。

安全规则：
- 只能**追加到文件末尾**（末尾之前的字节偏移不变，运行中的 bash 不受影响）；
- 中间任何位置的修改，必须等脚本跑完；
- Python 脚本没有这个问题（一次性读完编译），shell 才有。
