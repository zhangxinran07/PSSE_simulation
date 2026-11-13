# -*- coding: utf-8 -*-
"""B 线：真实 PSSE 大样本数据生成（分片 / 断点续跑 / 独立 case 目录）。

背景：B 线要验证 A2.31~A2.34 的 30 组结论（95%CI 覆盖率 9.6%、收缩比 0.769、
      H2_a↔H2_b 相关 −0.955、λ≈0.04 建议）在大样本上是否成立。标准入口
      Sample_Generation/main_simu.py:simu_iden_mix 每组都顺带做一次单次辨识，
      且固定输出文件名，不便分片；本脚本只做「仿真 + 落盘原始 u/P/Q + 真值」，
      辨识与拉普拉斯后验交给后续分析脚本。

隔离：每个分片用独立的 Cases/w<sid>/ 目录（只含 3 个必需 case 文件），避免
      .sav / .snp / .out / conec.for 等临时文件互相覆盖（C.6 的文件冲突根源）。
      是否允许单机多实例由 license 决定，先用 --probe 实测。

用法：
    python scripts/b_line_shard.py <shard_id> <n_groups> --probe   # 只跑 1 组测通断
    python scripts/b_line_shard.py <shard_id> <n_groups>            # 正式跑，逐组落盘可续跑

输出：Sample_Generation/results/b_line_ori_shard<sid>.pkl
      {'shard': sid, 'groups': [{'idx', 'u', 'P', 'Q', 'truth', 'simulation_time', 'wall_time'}, ...]}
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

# ⚠️ PSSE 必须在 numpy / matplotlib 之前 import（MKL DLL 冲突），与 main_simu.py 一致
import psse_dynamic as ps_d
import numpy as np
import main_simu as M

RES = os.path.join(ROOT, 'Sample_Generation', 'results')
CASE_SRC = os.path.join(ROOT, 'Cases', 'IEEE39')
CASE_FILES = ['ieee39_RE.sav', 'IEEE39_RE.dyr',
              'ieee39_RE_ieee39_RE_motorD_Stall_enabled.dyr']
BASE_SEED = 20261110
BUSID = 3


def prepare_case(sid):
    dst = os.path.join(ROOT, 'Cases', 'w%d' % sid)
    os.makedirs(dst, exist_ok=True)
    for f in CASE_FILES:
        s, d = os.path.join(CASE_SRC, f), os.path.join(dst, f)
        if not os.path.exists(d):
            shutil.copy2(s, d)
    # ⚠️ 实测：PSS/E 的 Fortran 文件打开处理不了含中文的绝对路径
    #    （psspy.case 报 ierr=3「File not found」，路径按本地编码乱码）。
    #    故每个分片进程 chdir 到自己的目录、case_path 固定用 './'：
    #    .sav/.dyr/.snp/.out/conec.for 全部落在各自目录，天然隔离，且全为相对路径。
    #    Python 侧（读 dyr / 写 pkl）用绝对路径没问题，另返回 dyr 绝对路径。
    os.chdir(dst)
    return './', os.path.join(dst, CASE_FILES[2])


def build_kwargs(case_path, dyr_abs):
    # ⚠️ 必须原地改 M.kwargs，不能复制：main_simu.sample_params 直接写模块级
    # kwargs['ld_con_chng_values'][index][...]，复制会导致 IndexError。
    kw = M.kwargs
    kw['case_path'] = case_path
    kw['load_bus_monitor'] = [BUSID]
    kw['ld_con_chng_buses'] = []
    kw['ld_con_chng_values'] = []
    for busid in M.busid_set:
        kw['ld_con_chng_buses'].append(busid)
        kw['ld_con_chng_values'].append(M.ps_d.read_bus_params(busid=busid, path=dyr_abs))
    return kw


def extract(df):
    cols = list(df.columns)

    def col(key):
        hit = df.values[:, [i for i in range(len(cols)) if (key % BUSID) in cols[i]]]
        return hit.reshape(len(df))

    ud, uq = col('BUS%d UD'), col('BUS%d UQ')
    P, Q = col('PLOD %d['), col('QLOD %d[')
    return ud + 1j * uq, P, Q


def run(sid, n_groups, probe=False):
    case_path, dyr_abs = prepare_case(sid)
    kw = build_kwargs(case_path, dyr_abs)
    out = os.path.join(RES, 'b_line_ori_shard%d.pkl' % sid)
    groups = []
    if not probe and os.path.exists(out):
        groups = pickle.load(open(out, 'rb'))['groups']
        print('[shard %d] 断点续跑, 已有 %d 组' % (sid, len(groups)), flush=True)
    # 每个分片独立 seed，保证可复现且组间真值不重复
    np.random.seed(BASE_SEED + sid * 100003)
    t_start = time.time()
    for k in range(len(groups), n_groups):
        kw['ambient_seed'] = int(np.random.randint(0, 2 ** 31 - 1))
        truth = np.asarray(M.sample_params(BUSID, 'mix'), dtype=float)
        t0 = time.time()
        df, d_para, st = ps_d.one_time_simulation(kw, max_retries=5)
        if df is None:
            print('[shard %d] group %d 仿真失败, 跳过' % (sid, k), flush=True)
            continue
        u, P, Q = extract(df)
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
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    run(sid, n, probe=('--probe' in sys.argv))
