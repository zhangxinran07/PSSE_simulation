# -*- coding: utf-8 -*-
"""换母线对齐实验：bus3 vs bus4（扰动电压幅值对齐到 bus3 一万组中位 0.0218 p.u.）。

背景（张老师 2026-11-13 要求）：
  实际辨识落点可能在任意连着负荷模型的母线，换母线扫描是必要工作；且对比时须保证
  「扰动幅值相同」——同样的功率扰动加在不同母线，得到的电压扰动幅值不同，否则不公平。
  （上一轮 A2.45 用同 500MW 扰动对比 bus3/bus4，实测两者电压幅值恰好接近 <1%，属运气。）

本实验设计：
  · 对齐锚点 = bus3 主线一万组(10000 组, A2.41)逐组实测电压扰动幅值 |u| max-min 中位
    = 0.0218 p.u.（均值 0.0221, std 0.0036，实测 16 个分片 10000 组）。
  · bus3：保持主线一万组口径 ambient_a 500MW（电压幅值中位天然 0.0218），零改动。
  · bus4：ambient_a 固定 MW_target（由 bus4 标定曲线插值反解，使其电压幅值中位同样 = 0.0218）。
    两者同为 ambient_a 固定 MW，仅 MW 值不同（因电气位置不同），电压扰动幅值分布对称对齐。
  · 同 seed、同真值，唯一变量 = 母线位置（电压扰动幅值已对齐）。

用法（两个阶段）：
  python scripts/bus_switch_align.py gen <busid> <n> <mw> [--seed S] [--tag NAME] [--probe]
  python scripts/bus_switch_align.py analyze <n> [K]

产出：
  Sample_Generation/results/<tag>_ori.pkl   （生成数据）
  Sample_Generation/results/bus_align_probe.txt  （辨识对比 + 幅值对齐验证 + wilcoxon 配对）
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
from scipy.stats import wilcoxon
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
BASE_SEED = 20261112        # 辨识多起点种子基
TARGET_U = 0.0218           # bus3 一万组电压扰动幅值中位（对齐锚点）


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


def gen(busid, n_groups, seed, tag, mw, probe=False):
    sid = busid * 10 + (1 if busid == 4 else 0)  # bus3→w30, bus4→w41
    case_path, dyr_abs = prepare_case(sid)
    kw = build_kwargs(case_path, dyr_abs, busid)
    kw['disturbance_type'] = 'ambient_a'
    kw['disturbance_range'] = mw          # 固定 MW（bus3=500, bus4=对齐反解值）
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
        pickle.dump({'busid': busid, 'mw': mw, 'groups': groups}, open(out, 'wb'))
        print('[gen bus%d] %d/%d 完成 (%.1fs), 累计 %d 组' % (
            busid, k + 1, n_groups, time.time() - t0, len(groups)), flush=True)
    print('[gen bus%d] DONE %d 组, 用时 %.1f min' % (
        busid, len(groups), (time.time() - t_start) / 60), flush=True)
    return True


def load_pkl(path):
    return pickle.load(open(path, 'rb'))['groups']


def align(g3, g4):
    """按组序号（idx % 100000）对齐 bus3/bus4，取两者都有的组。"""
    d3 = {g['idx'] % 100000: g for g in g3}
    d4 = {g['idx'] % 100000: g for g in g4}
    keys = sorted(set(d3) & set(d4))
    return [d3[k] for k in keys], [d4[k] for k in keys]


def urange(g):
    return float(np.abs(g['u']).max() - np.abs(g['u']).min())


def identify_groups(groups, K, seed_base):
    recs = []
    for g in groups:
        th = identify_mix(g['u'], g['P'], g['Q'], 0.01, lam_H=LAM_H, K=K, seed=seed_base + g['idx'])
        recs.append((np.asarray(th.x, float), np.asarray(g['truth'], float)))
    return recs


def analyze(n, K):
    g3 = load_pkl(os.path.join(RES, 'bus3_align_ori.pkl'))
    g4 = load_pkl(os.path.join(RES, 'bus4_align_ori.pkl'))
    g3, g4 = align(g3, g4)
    g3, g4 = g3[:n], g4[:n]
    print('[analyze] 对齐后 bus3=%d 组, bus4=%d 组（取交集）' % (len(g3), len(g4)), flush=True)

    tr3 = np.array([np.asarray(g['truth'], float) for g in g3])
    tr4 = np.array([np.asarray(g['truth'], float) for g in g4])
    print('[analyze] 真值一致校验 max|tr3-tr4| = %.3e（应≈0）' % np.max(np.abs(tr3 - tr4)), flush=True)

    # 幅值对齐验证
    r3 = np.array([urange(g) for g in g3])
    r4 = np.array([urange(g) for g in g4])

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
    dE = E4 - E3

    L = []
    ap = L.append
    ap('=' * 96)
    ap('换母线对齐实验：bus3 vs bus4（扰动电压幅值对齐到 bus3 一万组中位 %.4f p.u.，%d 组配对, K=%d, λ_H=%.2f）'
       % (TARGET_U, len(g3), K, LAM_H))
    ap('=' * 96)
    ap('')
    ap('[0] 扰动幅值对齐验证（|u| max-min, p.u.）')
    ap('  %-8s %9s %9s %9s %9s %9s' % ('母线', '中位', '均值', 'std', 'min', 'max'))
    ap('  %-8s %9.4f %9.4f %9.4f %9.4f %9.4f' % (
        'bus3', np.median(r3), r3.mean(), r3.std(), r3.min(), r3.max()))
    ap('  %-8s %9.4f %9.4f %9.4f %9.4f %9.4f' % (
        'bus4', np.median(r4), r4.mean(), r4.std(), r4.min(), r4.max()))
    ap('  中位比 bus4/bus3 = %.3f（应≈1.00）' % (np.median(r4) / np.median(r3)))
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
    ap('[3] 逐参数配对 wilcoxon（正=bus4 更差；p<0.05 显著）')
    ap('  %-8s %9s %9s %9s %9s %11s %8s' % ('参数', 'bus3中位', 'bus4中位', 'Δ中位(pp)', '同向%', 'wilcoxon p', '显著'))
    for j, p in enumerate(PNAMES):
        d = dE[:, j]
        same = np.mean(d > 0) * 100
        try:
            _, pv = wilcoxon(E4[:, j], E3[:, j])
        except ValueError:
            pv = float('nan')
        sig = '**' if pv < 0.01 else ('*' if pv < 0.05 else '')
        ap('  %-8s %9.2f %9.2f %+9.2f %9.1f %11.4f %8s' % (
            p, np.median(E3[:, j]), np.median(E4[:, j]), np.median(d), same, pv, sig))
    ap('')
    ap('[4] 汇总')
    ap('  ΔP_im=%+.2fpp  ΔH2=%+.2fpp  Δdyn6=%+.2fpp' % (
        np.median(dE[:, PIM]), np.median(dE[:, H2]), np.median(dE[:, DYN])))
    ap('=' * 96)

    txt = '\n'.join(L)
    out = os.path.join(RES, 'bus_align_probe.txt')
    open(out, 'w', encoding='utf-8').write(txt + '\n')
    print('\n' + txt, flush=True)
    print('[saved] %s' % out, flush=True)

    # 落盘 th/truth 数组供后续复查
    pickle.dump(dict(th3=th3, tr3=tr3, th4=th4, tr4=tr4, r3=r3, r4=r4),
                open(os.path.join(RES, 'bus_align_arrays.pkl'), 'wb'))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'analyze'
    if cmd == 'gen':
        busid = int(sys.argv[2])
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        mw = float(sys.argv[4]) if len(sys.argv) > 4 else (500.0 if busid == 3 else 472.0)
        seed = 20261113
        tag = 'bus%d_align' % busid
        probe = False
        args = sys.argv[5:]
        for i, a in enumerate(args):
            if a == '--seed':
                seed = int(args[i + 1])
            elif a == '--tag':
                tag = args[i + 1]
            elif a == '--probe':
                probe = True
        gen(busid, n, seed, tag, mw, probe)
    else:
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        K = int(sys.argv[3]) if len(sys.argv) > 3 else K_START
        analyze(n, K)
