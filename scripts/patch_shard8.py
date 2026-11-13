# -*- coding: utf-8 -*-
"""shard8 补跑：跳过卡死组 idx=5600018，其余组带单组超时保护。

背景：b10k_analyze.py solve 8/12 卡在 idx=5600018（Dell 数据）那一组，
根因是 identify() 里的 trust-constr 无 maxiter + while True 初值采样无上限，
偶发死循环。本脚本：
  1) 读回 shard8 已落盘的 730 组；
  2) 对剩余组（mine[730:]）逐组在独立子进程里辨识，单组超时 120s，
     超时/异常的组跳过并记录；
  3) 每 5 组落盘一次，最后写回 b10k_lap_shard8.pkl。
"""
import os
import sys
import time
import pickle
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'package'))
from identification import make_mix_functions, LAM_H_DEFAULT
from laplace_posterior import one_group
import b10k_analyze as B

KEEP = B.KEEP
K_START = 12
EPS = 1e-4
TIMEOUT = 120  # 单组超时秒数
SKIP_IDX = {5600030}  # 发散组（P/Q 大负值、obj 恒 1e16），数据本身坏，剔除

OUT = os.path.join(B.RES, '%s8.pkl' % B.LAP_PREFIX)


def _work(args):
    """子进程工作函数（顶层可 pickle）。返回 (idx, ok, rec_or_err)。"""
    g, u, P, Q, truth, src = args
    try:
        obj, jac = make_mix_functions(LAM_H_DEFAULT)
        o = one_group(g, u, P, Q, truth, K_START, EPS, obj, jac)
        rec = {k: o[k] for k in KEEP}
        rec['src'] = src
        return (g, True, rec)
    except Exception as e:
        return (g, False, repr(e))


def main():
    groups = B.load_all()
    mine = groups[8::12]
    print('本片总组数 %d, 已落盘断点由文件决定' % len(mine), flush=True)

    recs = []
    if os.path.exists(OUT):
        recs = pickle.load(open(OUT, 'rb'))
    start = len(recs)
    print('断点续跑, 已有 %d 组, 从下标 %d 开始' % (start, start), flush=True)

    obj, jac = make_mix_functions(LAM_H_DEFAULT)  # 主进程也准备一份（备用）
    todo = mine[start:]
    skipped = []
    failed = []
    t0 = time.time()

    for i, r in enumerate(todo):
        g = int(r['idx'])
        if g in SKIP_IDX:
            skipped.append(g)
            print('  跳过卡死组 idx=%d' % g, flush=True)
            continue
        args = (g, r['u'], r['P'], r['Q'], r['truth'], r['src'])
        ctx = mp.get_context('spawn')
        q = ctx.Queue()
        p = ctx.Process(target=_run_and_put, args=(args, q))
        p.start()
        p.join(TIMEOUT)
        if p.is_alive():
            p.terminate()
            p.join()
            failed.append((g, 'timeout'))
            print('  idx=%d 超时(%ds), 跳过' % (g, TIMEOUT), flush=True)
            continue
        _, ok, res = q.get()
        if ok:
            recs.append(res)
        else:
            failed.append((g, res))
            print('  idx=%d 异常: %s' % (g, res), flush=True)

        if len(recs) % 5 == 0:
            pickle.dump(recs, open(OUT, 'wb'))
            print('  %d/%d 完成, 用时 %.1f min' % (
                len(recs), len(todo) + start, (time.time() - t0) / 60), flush=True)

    pickle.dump(recs, open(OUT, 'wb'))
    print('[patch shard8] DONE: 有效 %d 组, 跳过 %s, 失败 %s' % (
        len(recs), skipped, failed), flush=True)


def _run_and_put(args, q):
    res = _work(args)
    q.put(res)


if __name__ == '__main__':
    main()
