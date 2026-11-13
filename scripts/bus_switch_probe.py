# -*- coding: utf-8 -*-
"""换母线敏感性实验：bus3 vs bus4（同种子同真值，隔离母线位置变量）。

回答张老师「现在是 bus3，换到 bus4 试试」——检验辨识误差是否随观测母线位置变化：
  若 bus3/bus4 误差接近 → 结构失配是「模型结构」共性因素（纯电机 vs CMLD），与母线位置无关；
  若差异显著   → 网络耦合/电气距离影响辨识，可选更优观测母线（本身即改进线索）。

设计：同 np.random 种子生成 bus3、bus4 各 N 组。sample_params 采样值只依赖 np.random 状态、
  与 busid 无关（busid 只决定写入 ld_con_chng_values 的哪个 index）→ 两母线逐组真值完全相同，
  唯一变量 = 母线位置。辨识统一 K=12 起点 + λ_H=0.02 + 解析雅可比（identify_mix 主线入口）。

用法（两个阶段，分开跑，数据落盘后可反复 analyze）：
  python scripts/bus_switch_probe.py gen <busid> <n> [--seed S] [--tag NAME] [--probe]
  python scripts/bus_switch_probe.py analyze <n> [K]

输出：Sample_Generation/results/<tag>_ori.pkl（生成数据）
      Sample_Generation/results/bus_switch_probe.txt（辨识对比）
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
from identification import identify_mix

RES = os.path.join(ROOT, 'Sample_Generation', 'results')
CASE_SRC = os.path.join(ROOT, 'Cases', 'IEEE39')
CASE_FILES = ['ieee39_RE.sav', 'IEEE39_RE.dyr',
              'ieee39_RE_ieee39_RE_motorD_Stall_enabled.dyr']
PNAMES = ['a_a', 'b_a', 'H2_a', 'P_ima', 'a_b', 'b_b', 'H2_b', 'P_imb']
DYN = [0, 1, 2, 4, 5, 6]
PIM = [3, 7]
H2 = [2, 6]
LAM_H = 0.02
K_START = 12
BASE_SEED = 20261112


def prepare_case(sid):
    """复制 3 个 case 文件到隔离目录并 chdir（PSSE 的 Fortran 处理不了中文绝对路径）。"""
    dst = os.path.join(ROOT, 'Cases', 'w%d' % sid)
    os.makedirs(dst, exist_ok=True)
    for f in CASE_FILES:
        s, d = os.path.join(CASE_SRC, f), os.path.join(dst, f)
        if not os.path.exists(d):
            shutil.copy2(s, d)
    os.chdir(dst)
    return './', os.path.join(dst, CASE_FILES[2])


def build_kwargs(case_path, dyr_abs, busid):
    """就地配置模块级 M.kwargs（sample_params 直接写它，必须同一对象）。"""
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


def gen(busid, n_groups, seed, tag, probe=False):
    sid = busid * 10 + (1 if busid == 4 else 0)  # bus3→w30, bus4→w41，避开 w0..7
    case_path, dyr_abs = prepare_case(sid)
    kw = build_kwargs(case_path, dyr_abs, busid)
    out = os.path.join(RES, '%s_ori.pkl' % tag)
    groups = []
    if not probe and os.path.exists(out):
        groups = pickle.load(open(out, 'rb'))['groups']
        print('[gen bus%d] 断点续跑, 已有 %d 组' % (busid, len(groups)), flush=True)
    np.random.seed(seed)
    t_start = time.time()
    for k in range(len(groups), n_groups):
        kw['ambient_seed'] = int(np.random.randint(0, 2 ** 31 - 1))
        truth = np.asarray(M.sample_params(busid, 'mix'), dtype=float)
        t0 = time.time()
        df, d_para, st = ps_d.one_time_simulation(kw, max_retries=5)
        if df is None:
            print('[gen bus%d] group %d 仿真失败, 跳过' % (busid, k), flush=True)
            continue
        u, P, Q = extract_bus(df, busid)
        if probe:
            print('[gen bus%d] PROBE OK | wall %.1fs | sim %.1fs | len(u)=%d | truth=%s' % (
                busid, time.time() - t0, float(st), len(u),
                np.array2string(truth, precision=3)), flush=True)
            return True
        groups.append(dict(idx=busid * 100000 + k, u=u, P=P, Q=Q, truth=truth,
                           simulation_time=float(st), wall_time=time.time() - t0))
        pickle.dump({'busid': busid, 'groups': groups}, open(out, 'wb'))
        print('[gen bus%d] %d/%d 完成 (%.1fs), 累计 %d 组' % (
            busid, k + 1, n_groups, time.time() - t0, len(groups)), flush=True)
    print('[gen bus%d] DONE %d 组, 用时 %.1f min' % (
        busid, len(groups), (time.time() - t_start) / 60), flush=True)
    return True


def load_pkl(path):
    return pickle.load(open(path, 'rb'))['groups']


def align(g3, g4):
    """按组序号（idx % 100000）对齐 bus3/bus4，取两者都有的组（同种子下真值应逐组相同）。"""
    d3 = {g['idx'] % 100000: g for g in g3}
    d4 = {g['idx'] % 100000: g for g in g4}
    keys = sorted(set(d3) & set(d4))
    return [d3[k] for k in keys], [d4[k] for k in keys]


def identify_groups(groups, K, seed_base):
    recs = []
    for g in groups:
        u, P, Q = g['u'], g['P'], g['Q']
        truth = np.asarray(g['truth'], dtype=float)
        best = identify_mix(u, P, Q, 0.01, lam_H=LAM_H, K=K, seed=seed_base + g['idx'])
        th = np.asarray(best.x, dtype=float)
        recs.append((th, truth))
    return recs


def analyze(n, K):
    g3 = load_pkl(os.path.join(RES, 'bus3_sw_ori.pkl'))
    g4 = load_pkl(os.path.join(RES, 'bus4_sw_ori.pkl'))
    g3, g4 = align(g3, g4)
    g3, g4 = g3[:n], g4[:n]
    print('[analyze] 对齐后 bus3=%d 组, bus4=%d 组（取交集）' % (len(g3), len(g4)), flush=True)

    # 验证同种子下两母线真值逐组一致（应 np.allclose）
    tr3 = np.array([np.asarray(g['truth'], float) for g in g3])
    tr4 = np.array([np.asarray(g['truth'], float) for g in g4])
    print('[analyze] 真值一致校验 max|tr3-tr4| = %.3e（应≈0）' % np.max(np.abs(tr3 - tr4)), flush=True)

    t0 = time.time()
    rec3 = identify_groups(g3, K, BASE_SEED)
    print('[analyze] bus3 辨识完成 %.1fs' % (time.time() - t0), flush=True)
    t0 = time.time()
    rec4 = identify_groups(g4, K, BASE_SEED)
    print('[analyze] bus4 辨识完成 %.1fs' % (time.time() - t0), flush=True)

    th3 = np.array([r[0] for r in rec3])
    th4 = np.array([r[0] for r in rec4])
    E3 = np.abs(th3 - tr3) / np.abs(tr3) * 100
    E4 = np.abs(th4 - tr4) / np.abs(tr4) * 100

    L = []
    ap = L.append
    ap('=' * 96)
    ap('换母线敏感性实验：bus3 vs bus4（同种子同真值，%d 组配对, K=%d 起点, λ_H=%.2f）'
       % (len(g3), K, LAM_H))
    ap('=' * 96)
    ap('')
    ap('[1] 总体精度对比（中位误差 %%）')
    ap('  %-8s %9s %9s %9s' % ('母线', 'P_im%', 'H2%', 'dyn6%'))
    ap('  %-8s %9.2f %9.2f %9.2f' % ('bus3', np.median(E3[:, PIM]), np.median(E3[:, H2]), np.median(E3[:, DYN])))
    ap('  %-8s %9.2f %9.2f %9.2f' % ('bus4', np.median(E4[:, PIM]), np.median(E4[:, H2]), np.median(E4[:, DYN])))
    ap('')
    ap('[2] 逐参数中位误差（%%）')
    ap('  %-8s' % '母线' + ''.join(' %8s' % p for p in PNAMES))
    ap('  %-8s' % 'bus3' + ''.join(' %8.2f' % np.median(E3[:, j]) for j in range(8)))
    ap('  %-8s' % 'bus4' + ''.join(' %8.2f' % np.median(E4[:, j]) for j in range(8)))
    ap('')
    ap('[3] bus4 vs bus3 配对差异（pp，正=bus4 更差）')
    dE = E4 - E3
    ap('  ΔP_im=%+.2fpp   ΔH2=%+.2fpp   Δdyn6=%+.2fpp' % (
        np.median(dE[:, PIM]), np.median(dE[:, H2]), np.median(dE[:, DYN])))
    ap('  逐参数：' + ''.join(' %s%+.2f' % (p, np.median(dE[:, j])) for p, j in zip(PNAMES, range(8))))
    ap('')
    ap('[4] 关键判读')
    ap('  · 若 bus3/bus4 各项误差 |Δ| < 1~2pp（配对）→ 结构失配与母线位置无关，')
    ap('    是「纯电机 vs CMLD」共性因素主导，主线结论普适、不依赖 bus3 特例；')
    ap('  · 若某母线明显更差 → 网络耦合/电气距离是重要来源，可筛选更优观测母线。')
    ap('=' * 96)

    txt = '\n'.join(L)
    out = os.path.join(RES, 'bus_switch_probe.txt')
    open(out, 'w', encoding='utf-8').write(txt + '\n')
    print('\n' + txt, flush=True)
    print('[saved] %s' % out, flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'analyze'
    if cmd == 'gen':
        busid = int(sys.argv[2])
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 30
        seed = BASE_SEED
        tag = 'bus%d_sw' % busid
        probe = False
        args = sys.argv[4:]
        for i, a in enumerate(args):
            if a == '--seed':
                seed = int(args[i + 1])
            elif a == '--tag':
                tag = args[i + 1]
            elif a == '--probe':
                probe = True
        gen(busid, n, seed, tag, probe)
    else:
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        K = int(sys.argv[3]) if len(sys.argv) > 3 else K_START
        analyze(n, K)
