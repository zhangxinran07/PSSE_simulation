# -*- coding: utf-8 -*-
"""Dell 补跑 1 组：重新仿真 idx=5600030（shard6 第 30 组，原数据仿真发散）。

背景：b5k_dell_ori_shard6.pkl 里 idx=5600030 这一组 P/Q 出现大负值、电压实部飙到 5.63，
是 PSSE 仿真数值发散产生的坏数据（不是辨识失败）。本脚本在 Dell 上用新随机种子重新
仿真这一组，用干净数据覆盖原坏数据（truth 也重新采样），凑满 10000 有效组。

用法（Dell Tower, psse_39 环境）：
    C:/Users/ADMIN/anaconda3/envs/psse_39/python.exe scripts/b5k_dell_refix_one.py
"""
import os
import sys
import time
import pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)

import b_line_shard as BS

RES = BS.RES
TARGET_IDX = 5600030
SID = 6
SHARD_FILE = os.path.join(RES, 'b5k_dell_ori_shard6.pkl')


def main():
    data = pickle.load(open(SHARD_FILE, 'rb'))
    groups = data['groups']
    # 定位坏组
    pos = None
    for i, g in enumerate(groups):
        if g['idx'] == TARGET_IDX:
            pos = i
            break
    if pos is None:
        print('未找到 idx=%d, 退出' % TARGET_IDX)
        return
    old = groups[pos]
    print('定位坏组 pos=%d, 原 truth=%s' % (pos, np.array2string(old['truth'], precision=3)))
    print('  原 P range [%.3g, %.3g], Q range [%.3g, %.3g]' % (
        float(np.min(old['P'])), float(np.max(old['P'])),
        float(np.min(old['Q'])), float(np.max(old['Q']))))

    # 复用 shard6 的 case 目录，重新仿真
    from b5k_dell_shard import dell_prepare_case
    case_path, dyr_abs = dell_prepare_case(SID)
    kw = BS.build_kwargs(case_path, dyr_abs)

    # 用全新随机数（不复用原 seed 序列，确保得到干净数据）
    np.random.seed(20261113 + TARGET_IDX)
    ok = False
    for attempt in range(10):
        kw['ambient_seed'] = int(np.random.randint(0, 2 ** 31 - 1))
        truth = np.asarray(BS.M.sample_params(BS.BUSID, 'mix'), dtype=float)
        t0 = time.time()
        df, d_para, st = BS.ps_d.one_time_simulation(kw, max_retries=5)
        if df is None:
            print('  attempt %d 仿真失败, 重试' % attempt)
            continue
        u, P, Q = BS.extract(df)
        # 校验数据干净（P/Q 无大负值、电压实部标幺合理）
        P = np.asarray(P); Q = np.asarray(Q); u = np.asarray(u)
        umax = u.real.max() if np.iscomplexobj(u) else u.max()
        clean = (P.min() > 0 and Q.min() > 0 and umax < 1.5
                 and P.max() < 100 and Q.max() < 100)
        print('  attempt %d: P[%.3g,%.3g] Q[%.3g,%.3g] umax=%.3g clean=%s wall=%.1fs' % (
            attempt, P.min(), P.max(), Q.min(), Q.max(), umax, clean, time.time() - t0))
        if clean:
            groups[pos] = dict(idx=TARGET_IDX, u=u, P=P, Q=Q, truth=truth,
                               simulation_time=float(st), wall_time=time.time() - t0,
                               _refix_note='refixed from diverged simulation')
            pickle.dump({'shard': SID, 'groups': groups}, open(SHARD_FILE, 'wb'))
            print('✅ 已覆盖 idx=%d 为干净数据, 新 truth=%s' % (
                TARGET_IDX, np.array2string(truth, precision=3)))
            ok = True
            break
    if not ok:
        print('❌ 10 次尝试都未得到干净数据')
        return

    # 校验 shard6 组数
    data2 = pickle.load(open(SHARD_FILE, 'rb'))
    print('shard6 组数 = %d (应 625)' % len(data2['groups']))


if __name__ == '__main__':
    main()
