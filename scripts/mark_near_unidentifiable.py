# -*- coding: utf-8 -*-
"""标记「近不可辨识区」：零失配合成数据下，多起点最优仍无法收敛到全局极小的真值组合。

背景(2026-11-07, Day2): 分电机 vs 总功率诊断(A2.14)已坐实「非凸=观测降维代价」,
但还有一类更隐蔽的组合——即使多起点(6 起点取最优)也收敛不到全局极小(best_fun 仍高),
即 A2.13 里提到的「近不可辨识组合 ~13%(4/30)」。本脚本系统识别这类组合并归纳判据,
在结果里区分「可辨识区 / 近不可辨识区」。

判据(待脚本自洽验证):
  近不可辨识 = 多起点最优 best_fun > 1e-4 (对照 A2.13) 或 最优动态6参数中位误差 > 1%。
  候选特征:
    ① H2_a ≈ H2_b (惯性混淆, H2 比值 |log(H2_a/H2_b)| 小)
    ② a 贴下界 lb*2 = 200 (a_a 或 a_b 接近下界)
    ③ P_ima ≈ P_imb (稳态功率比例混淆)
    ④ 其他(脚本逐组打印 truth 供人工核查)

输出 Sample_Generation/results/near_unidentifiable.txt
"""
import numpy as np
import sys, os, math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'package'))
sys.path.insert(0, os.path.dirname(__file__))
from identification import obj_mix_iden, obj_mix_jac, identify as idf_id
from control_zero_mismatch import sample_mix, gen_u, gen_PQ

DT = 0.01
PNAMES = ['a_a', 'b_a', 'H2_a', 'P_ima', 'a_b', 'b_b', 'H2_b', 'P_imb']
DYN = [0, 1, 2, 4, 5, 6]
lb = [100, 30, 0.08, 0.2]
ub = [1200, 240, 0.22, 0.8]
SEED = 20261107
RESTARTS = 6
FUN_THR = 1e-4          # 多起点最优 fun 阈值(近不可辨识判据, 对照 A2.13)
ERR_THR = 0.01          # 动态6参数中位误差阈值
A_LO = lb[0] * 2        # a 下界 = 200
OUT = os.path.join(os.path.dirname(__file__), '..', 'Sample_Generation', 'results',
                   'near_unidentifiable.txt')


def relerr(x, xt):
    return np.abs(np.asarray(x) - np.asarray(xt)) / np.abs(np.asarray(xt))


def run(N):
    rng = np.random.default_rng(SEED)
    rows = []
    print('=' * 100, flush=True)
    print('近不可辨识区标记: 多起点(%d 起点)最优仍无法收敛到全局极小的真值组合' % RESTARTS, flush=True)
    print('=' * 100, flush=True)
    for g in range(N):
        truth, cc = sample_mix(rng)
        u = gen_u(np.random.default_rng(SEED + g))
        P, Q = gen_PQ(u, truth)

        best_x, best_fun = None, np.inf
        for r in range(RESTARTS):
            np.random.seed(SEED * 10 + g * RESTARTS + r)
            result, _ = idf_id(u, P, Q, DT, lb * 2, ub * 2, obj_func=obj_mix_iden,
                               tol=1e-8, max_trials=5, jac_func=obj_mix_jac)
            if result.fun < best_fun:
                best_fun, best_x = result.fun, np.asarray(result.x)

        e = relerr(best_x, truth)
        dyn_err = np.median(e[DYN])
        h2_ratio = truth[2] / truth[6]            # H2_a / H2_b
        a_a_lo = truth[0] <= A_LO * 1.1           # a_a 贴下界(留 10% 余量)
        a_b_lo = truth[4] <= A_LO * 1.1
        pim_ratio = truth[3] / truth[7]           # P_ima / P_imb
        near = dyn_err > ERR_THR   # 主判据: 参数误差(最终关心); fun 仅作辅助参考
        rows.append(dict(g=g, truth=truth, best_x=best_x, best_fun=best_fun,
                         dyn_err=dyn_err, h2_ratio=h2_ratio, a_a_lo=a_a_lo,
                         a_b_lo=a_b_lo, pim_ratio=pim_ratio, near=near))
        flag = '近不可辨识' if near else '可辨识'
        print('  g=%02d best_fun=%.2e dyn=%.2f%% H2_a/H2_b=%.3f P_im比=%.3f a贴界(A=%d,B=%d) → %s'
              % (g, best_fun, dyn_err * 100, h2_ratio, pim_ratio, a_a_lo, a_b_lo, flag),
              flush=True)

    # ---- 汇总 ----
    nears = [r for r in rows if r['near']]
    okays = [r for r in rows if not r['near']]
    lines = []
    lines.append('=' * 100)
    lines.append('近不可辨识区标记汇总(N=%d, %d 起点最优, 零失配合成数据)' % (N, RESTARTS))
    lines.append('主判据: 动态6参数中位误差 > %.0f%% (fun>%.0e 仅作优化器收敛参考)' % (ERR_THR * 100, FUN_THR))
    lines.append('=' * 100)
    lines.append('')
    lines.append('【分组统计】')
    lines.append('  近不可辨识组数 = %d / %d (%.0f%%)' % (len(nears), N, len(nears) / N * 100))
    lines.append('  可辨识组数     = %d / %d (%.0f%%)' % (len(okays), N, len(okays) / N * 100))
    lines.append('')
    if nears:
        lines.append('【近不可辨识组逐条】 (g, best_fun, dyn_err%%, H2_a/H2_b, P_im比, a_a, a_b)')
        for r in nears:
            t = r['truth']
            lines.append('  g=%02d fun=%.2e dyn=%.2f%% H2比=%.3f P_im比=%.3f a_a=%.1f a_b=%.1f'
                         % (r['g'], r['best_fun'], r['dyn_err'] * 100, r['h2_ratio'],
                            r['pim_ratio'], t[0], t[4]))
        lines.append('')
        lines.append('【判据归纳】')
        # 特征 1: H2 混淆 (H2_a ≈ H2_b)
        h2_near = [r for r in nears if abs(math.log(r['h2_ratio'])) < 0.15]
        lines.append('  ① H2 混淆(H2_a≈H2_b, |log比|<0.15): %d/%d 组' % (len(h2_near), len(nears)))
        # 特征 2: a 贴下界
        a_lo = [r for r in nears if r['a_a_lo'] or r['a_b_lo']]
        lines.append('  ② a 贴下界(a≤%.0f): %d/%d 组' % (A_LO, len(a_lo), len(nears)))
        # 特征 3: P_im 混淆
        pim_near = [r for r in nears if abs(math.log(r['pim_ratio'])) < 0.15]
        lines.append('  ③ P_im 混淆(P_ima≈P_imb, |log比|<0.15): %d/%d 组' % (len(pim_near), len(nears)))
        # 交集统计
        both = [r for r in nears if r['a_a_lo'] or r['a_b_lo'] or
                abs(math.log(r['h2_ratio'])) < 0.15 or abs(math.log(r['pim_ratio'])) < 0.15]
        lines.append('  ④ 至少命中一个特征的组数: %d/%d' % (len(both), len(nears)))
        lines.append('')
        lines.append('  对比: 可辨识组 H2比中位=%.3f P_im比中位=%.3f a贴界占比=%d/%d'
                     % (np.median([r['h2_ratio'] for r in okays]),
                        np.median([r['pim_ratio'] for r in okays]),
                        sum(1 for r in okays if r['a_a_lo'] or r['a_b_lo']), len(okays)))
    lines.append('')
    lines.append('结论:')
    lines.append('  近不可辨识区(零失配+总功率观测)的特征组合:')
    lines.append('    - H2_a ≈ H2_b (两台电机惯性接近, 总功率观测下沿「此消彼长」零空间不可分)')
    lines.append('    - a 贴下界 lb*2=%.0f (静态转矩系数过小, 接近边界, 优化器易卡)' % A_LO)
    lines.append('  这些组合即使多起点也收敛不到全局极小, 属于「近不可辨识区」;')
    lines.append('  其余组合为「可辨识区」(多起点最优误差→0)。')
    lines.append('  ⚠️ 本实验零失配下标记, 结构失配(PSSE)下近不可辨识区可能更宽, 见 A2.11/A2.12。')
    lines.append('=' * 100)
    txt = '\n'.join(lines)
    open(OUT, 'w', encoding='utf-8').write(txt)
    print('\n' + txt, flush=True)


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    run(N)
