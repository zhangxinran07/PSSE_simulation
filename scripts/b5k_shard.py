# -*- coding: utf-8 -*-
"""海量 5000 组：全新 PSSE 仿真数据生成（复用 b_line_shard 的隔离框架）。

与 b_line_shard.py（1000 组，A2.37）的区别：
  - BASE_SEED = 20261111（新种子，确保真值采样与 ambient_seed 与 1000 组不重复）
  - 输出 b5k_ori_shard<sid>.pkl（新文件名前缀，不覆盖 1000 组的 b_line_ori_*）
  - 默认每片 625 组（8 片 × 625 = 5000）

每一次都是 PSSE 真跑：每组 `ps_d.one_time_simulation(kw)` 重新仿真（ambient_seed 随机、
真值 sample_params 随机），绝不复用历史 u/P/Q。复用 b_line_shard 的 prepare_case（独立
case 目录 + chdir 隔离）/ build_kwargs / extract。

用法：
    python scripts/b5k_shard.py <shard_id> <n_groups> --probe   # 只跑 1 组测通断
    python scripts/b5k_shard.py <shard_id> <n_groups>            # 正式跑，逐组落盘可续跑

输出：Sample_Generation/results/b5k_ori_shard<sid>.pkl
      {'shard': sid, 'groups': [{'idx','u','P','Q','truth','simulation_time','wall_time'}, ...]}
"""
import os
import sys
import time
import pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)

import b_line_shard as BS   # 复用 prepare_case / build_kwargs / extract

RES = BS.RES
BASE_SEED = 20261111        # 全新种子（1000 组用 20261110）
PREFIX = 'b5k_ori_shard'    # 新前缀，不覆盖 b_line_ori_shard*


def run(sid, n_groups, probe=False):
    case_path, dyr_abs = BS.prepare_case(sid)
    kw = BS.build_kwargs(case_path, dyr_abs)
    out = os.path.join(RES, '%s%d.pkl' % (PREFIX, sid))
    groups = []
    if not probe and os.path.exists(out):
        groups = pickle.load(open(out, 'rb'))['groups']
        print('[shard %d] 断点续跑, 已有 %d 组' % (sid, len(groups)), flush=True)
    # 每个分片独立 seed，保证可复现且组间真值不重复
    np.random.seed(BASE_SEED + sid * 100003)
    t_start = time.time()
    for k in range(len(groups), n_groups):
        kw['ambient_seed'] = int(np.random.randint(0, 2 ** 31 - 1))
        truth = np.asarray(BS.M.sample_params(BS.BUSID, 'mix'), dtype=float)
        t0 = time.time()
        df, d_para, st = BS.ps_d.one_time_simulation(kw, max_retries=5)
        if df is None:
            print('[shard %d] group %d 仿真失败, 跳过' % (sid, k), flush=True)
            continue
        u, P, Q = BS.extract(df)
        if probe:
            print('[shard %d] PROBE OK | wall %.1fs | sim_time %.1fs | len(u)=%d | truth=%s' % (
                sid, time.time() - t0, float(st), len(u),
                np.array2string(truth, precision=3)), flush=True)
            return True
        groups.append(dict(idx=sid * 100000 + k, u=u, P=P, Q=Q, truth=truth,
                           simulation_time=float(st), wall_time=time.time() - t0))
        pickle.dump({'shard': sid, 'groups': groups}, open(out, 'wb'))
        print('[shard %d] %d/%d 完成 (%.1fs), 累计 %d 组' % (
            sid, k + 1, n_groups, time.time() - t0, len(groups)), flush=True)
    print('[shard %d] DONE %d 组, 总用时 %.1f min' % (
        sid, len(groups), (time.time() - t_start) / 60), flush=True)
    return True


if __name__ == '__main__':
    sid = int(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 625
    run(sid, n, probe=('--probe' in sys.argv))
