# -*- coding: utf-8 -*-
"""分片生成 bus3/bus4 对齐实验数据（并行加速）+ 合并。

背景：单组仿真约 90s，100 组 × 2 母线单线程约 5 小时。本脚本把每母线拆 n_shards 片，
每片独立 case 目录（sid = busid*100 + shard）并行跑，最后 merge 成完整 pkl。

对齐口径：
  bus3 ambient_a 500 MW（电压幅值中位天然 0.0218 p.u.）
  bus4 ambient_a 550 MW（标定对齐值，见 bus4_calib.txt：urange=4.1e-5*MW-6.75e-4）

同 seed 同真值保证：bus3/bus4 的片 s 用相同 seed_base+shard*1000 → 真值逐组相同；
idx = busid*100000 + start + k（start = shard*per）→ align 用 idx%100000 逐组配对。

用法：
  python scripts/bus_align_shard.py gen <busid> <n_total> <mw> <shard> <n_shards> [tag]
  python scripts/bus_align_shard.py merge <busid> <n_shards> [tag]
  tag 默认 'bus%d_align'（输出 bus%d_align_s%d_ori.pkl / bus%d_align_ori.pkl）;
  传自定义 tag（如 'bus4_palign'）可生成独立命名的数据，避免覆盖既有对齐数据。
"""
import os
import sys
import time
import pickle
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'Sample_Generation'))
sys.path.insert(0, os.path.join(ROOT, 'package'))

import psse_dynamic as ps_d   # 必须先于 numpy import（PSSE 与 MKL DLL 冲突）
import numpy as np
import main_simu as M

RES = os.path.join(ROOT, 'Sample_Generation', 'results')
CASE_SRC = os.path.join(ROOT, 'Cases', 'IEEE39')
CASE_FILES = ['ieee39_RE.sav', 'IEEE39_RE.dyr',
              'ieee39_RE_ieee39_RE_motorD_Stall_enabled.dyr']
BASE_SEED = 20261113


def prepare_case(sid):
    dst = os.path.join(ROOT, 'Cases', 'w%d' % sid)
    os.makedirs(dst, exist_ok=True)
    for f in CASE_FILES:
        s, d = os.path.join(CASE_SRC, f), os.path.join(dst, f)
        if not os.path.exists(d):
            shutil.copy2(s, d)
    os.chdir(dst)
    return './', os.path.join(dst, CASE_FILES[2])


def build_kwargs(case_path, dyr_abs, busid):
    kw = M.kwargs
    kw['case_path'] = case_path
    kw['load_bus_monitor'] = [busid]
    kw['ld_con_chng_buses'] = []
    kw['ld_con_chng_values'] = []
    for b in M.busid_set:
        kw['ld_con_chng_buses'].append(b)
        kw['ld_con_chng_values'].append(M.ps_d.read_bus_params(busid=b, path=dyr_abs))
    return kw


def extract_bus(df, busid):
    cols = list(df.columns)

    def col(key):
        hit = df.values[:, [i for i in range(len(cols)) if (key % busid) in cols[i]]]
        return hit.reshape(len(df))

    ud, uq = col('BUS%d UD'), col('BUS%d UQ')
    P, Q = col('PLOD %d['), col('QLOD %d[')
    return ud + 1j * uq, P, Q


def gen(busid, n_total, mw, shard, n_shards, tag=None):
    per = n_total // n_shards
    start = shard * per
    sid = busid * 100 + shard                       # 独立 case 目录，并行不冲突
    case_path, dyr_abs = prepare_case(sid)
    kw = build_kwargs(case_path, dyr_abs, busid)
    kw['disturbance_type'] = 'ambient_a'
    kw['disturbance_range'] = mw
    if tag is None:
        tag = 'bus%d_align_s%d' % (busid, shard)
    else:
        tag = '%s_s%d' % (tag, shard)
    out = os.path.join(RES, '%s_ori.pkl' % tag)
    groups = []
    if os.path.exists(out):
        groups = pickle.load(open(out, 'rb'))['groups']
        print('[bus%d s%d] 断点续跑, 已有 %d 组' % (busid, shard, len(groups)), flush=True)
    np.random.seed(BASE_SEED + shard * 1000)        # bus3/bus4 同片同 seed → 真值相同
    for k in range(len(groups), per):
        kw['ambient_seed'] = int(np.random.randint(0, 2 ** 31 - 1))
        truth = np.asarray(M.sample_params(busid, 'mix'), dtype=float)
        t0 = time.time()
        df, d_para, st = ps_d.one_time_simulation(kw, max_retries=5)
        if df is None:
            print('[bus%d s%d] group %d 仿真失败, 跳过' % (busid, shard, k), flush=True)
            continue
        u, P, Q = extract_bus(df, busid)
        idx = busid * 100000 + start + k
        groups.append(dict(idx=idx, u=u, P=P, Q=Q, truth=truth,
                           simulation_time=float(st), wall_time=time.time() - t0))
        pickle.dump({'busid': busid, 'mw': mw, 'groups': groups}, open(out, 'wb'))
        print('[bus%d s%d] %d/%d 完成 (%.1fs)' % (
            busid, shard, k + 1, per, time.time() - t0), flush=True)
    print('[bus%d s%d] DONE %d 组' % (busid, shard, len(groups)), flush=True)


def merge(busid, n_shards, tag=None):
    if tag is None:
        tag = 'bus%d_align' % busid
    allg = []
    for s in range(n_shards):
        st = '%s_s%d' % (tag, s)
        p = os.path.join(RES, '%s_ori.pkl' % st)
        if not os.path.exists(p):
            print('[merge] 缺 %s' % p, flush=True)
            continue
        allg += pickle.load(open(p, 'rb'))['groups']
    allg.sort(key=lambda g: g['idx'])
    out = os.path.join(RES, '%s_ori.pkl' % tag)
    pickle.dump({'busid': busid, 'groups': allg}, open(out, 'wb'))
    print('[merge] bus%d 共 %d 组 -> %s' % (busid, len(allg), out), flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'gen':
        busid = int(sys.argv[2])
        n_total = int(sys.argv[3])
        mw = float(sys.argv[4])
        shard = int(sys.argv[5])
        n_shards = int(sys.argv[6])
        tag = sys.argv[7] if len(sys.argv) > 7 else None
        gen(busid, n_total, mw, shard, n_shards, tag)
    elif cmd == 'merge':
        busid = int(sys.argv[2])
        n_shards = int(sys.argv[3])
        tag = sys.argv[4] if len(sys.argv) > 4 else None
        merge(busid, n_shards, tag)
