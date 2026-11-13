# -*- coding: utf-8 -*-
"""Dell 补跑：shard3/5 各补 1 组，凑满 5000（本机 5000 + Dell 5000 = 10000）。

复用 b5k_dell_shard 的 dell_prepare_case / build_kwargs / extract，每组真跑一次 PSSE。
补跑的 idx 用 sid*100000 + 900 + k（原 shard3/5 的 k < 625，900 起保证唯一不冲突）。
失败自动换新 seed 重试（最多 20 次），直到补满。

用法（Dell 上 anaconda psse_39）：
    C:/Users/ADMIN/anaconda3/envs/psse_39/python.exe scripts/b5k_dell_patch.py
"""
import os
import sys
import time
import pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b5k_dell_shard as D


def patch(sid, n_extra=1):
    case_path, dyr_abs = D.dell_prepare_case(sid)
    kw = D.BS.build_kwargs(case_path, dyr_abs)
    out = os.path.join(D.RES, '%s%d.pkl' % (D.PREFIX, sid))
    groups = pickle.load(open(out, 'rb'))['groups']
    print('[patch shard%d] 原 %d 组, 需补 %d 组' % (sid, len(groups), n_extra), flush=True)
    np.random.seed(D.BASE_SEED + sid * 100003 + 77777)  # 独立于原 seed 序列
    added = 0
    attempts = 0
    while added < n_extra and attempts < 20:
        attempts += 1
        kw['ambient_seed'] = int(np.random.randint(0, 2 ** 31 - 1))
        truth = np.asarray(D.BS.M.sample_params(D.BS.BUSID, 'mix'), dtype=float)
        t0 = time.time()
        df, d_para, st = D.BS.ps_d.one_time_simulation(kw, max_retries=8)
        if df is None:
            print('[patch shard%d] 尝试 %d 失败, 换 seed 重试' % (sid, attempts), flush=True)
            continue
        u, P, Q = D.BS.extract(df)
        idx = D.IDX_OFFSET + sid * 100000 + 900 + added
        groups.append(dict(idx=idx, u=u, P=P, Q=Q, truth=truth,
                           simulation_time=float(st), wall_time=time.time() - t0))
        pickle.dump({'shard': sid, 'groups': groups}, open(out, 'wb'))
        added += 1
        print('[patch shard%d] 补成 1 组 (idx=%d), 累计 %d 组' % (sid, idx, len(groups)), flush=True)
    print('[patch shard%d] DONE, 补 %d 组, 共 %d 组' % (sid, added, len(groups)), flush=True)
    return added


if __name__ == '__main__':
    total = 0
    for sid in (3, 5):
        total += patch(sid, 1)
    print('[patch] 全部完成, 共补 %d 组' % total, flush=True)
