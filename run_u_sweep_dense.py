# run_u_sweep_dense.py - 电压幅值加密扫描调度(串行 + 断点续跑)
# 用途: 复用 run_u_sweep.py 逐档跑 ambient_u 0.03~0.08, 每档 30 组, 串行且跳过已完成档。
# 设计: 每档用独立 subprocess 调 run_u_sweep.py(每档独立时间戳结果文件), 避免同进程覆盖。
import os
import sys
import glob
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = r"C:\Users\Lenovo\.conda\envs\psse_312\python.exe"
SWEEP_SCRIPT = os.path.join(ROOT, "run_u_sweep.py")
RESULTS = os.path.join(ROOT, "Sample_Generation", "results")

U_RANGES = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
N_GROUPS = 30


def is_done(u):
    """该档是否已跑够 N_GROUPS 组。返回 (完成?, 匹配文件)。"""
    tag = "_u%.3f.pkl" % u
    pat = os.path.join(RESULTS, "result_all_mix_*" + tag)
    import pandas as pd
    best = None
    for f in sorted(glob.glob(pat)):
        try:
            df = pd.read_pickle(f)
            if len(df) >= N_GROUPS:
                return True, f
            best = (f, len(df))
        except Exception as e:
            print("读取失败 %s: %s" % (f, e), flush=True)
    return False, best


def main():
    print("=== 加密扫描启动: U=%s, N=%d ===" % (U_RANGES, N_GROUPS), flush=True)
    log = []
    for u in U_RANGES:
        done, info = is_done(u)
        if done:
            print("SKIP u=%.3f (已完成, %s)" % (u, info), flush=True)
            log.append((u, "skip"))
            continue
        print("=== START u=%.3f  (n=%d) ===" % (u, N_GROUPS), flush=True)
        r = subprocess.run(
            [PY, SWEEP_SCRIPT, "%.3f" % u, str(N_GROUPS)],
            cwd=ROOT,
        )
        if r.returncode == 0:
            print("=== DONE u=%.3f ===" % u, flush=True)
            log.append((u, "ok"))
        else:
            print("=== FAIL u=%.3f rc=%d ===" % (u, r.returncode), flush=True)
            log.append((u, "fail"))
            # 失败不中断, 继续下一档; 本档下次重跑会重试(断点续跑只跳过已完成档)
    print("=== ALL SWEEP LOG:", log, "===", flush=True)


if __name__ == "__main__":
    main()
