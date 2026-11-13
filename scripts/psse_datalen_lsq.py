# -*- coding: utf-8 -*-
"""数据长度(10s/30s/60s/120s)对辨识精度的影响 —— least_squares 口径。

张老师最新口径(2026-09-25):
  - 辨识器一律用 scipy.optimize.least_squares(method='trf', Gauss-Newton 内核), 不再用 trust-constr。
  - 采样约 10 组不同 con 参数真值, 每组跑一条 120s 真实 PSSE 仿真(12001 点),
    再截取前 10s/30s/60s/120s 四档分别辨识, 对比核心判据(P_im/H2)误差与 a/b 带符号偏差。

辨识器(least_squares)构造要点:
  - 原 obj_con_iden = sqrt(SS)/(n-k)*1000, SS 是 (P_p-P, Q_p-Q) 对 [1,V,V^2] 回归后的残差平方和。
  - least_squares 直接最小化逐点残差向量 r = concat(resid_P, resid_Q), sum(r^2)=SS,
    与原目标函数单调等价, 最优解一致。
  - 解析 Jacobian: _core_con_jac 给出 dP/dQ 对 (a,b,H2), 链式到 4 参数(a,b,H2,P_im)后
    再投影 (I-H) 得到残差 Jacobian。

用法:
    python psse_datalen_lsq.py sim [N_GROUPS]   # 只生成数据(120s x N 组, 分钟级)
    python psse_datalen_lsq.py iden             # 只读 pkl 跑四档辨识对比(分钟级)
    python psse_datalen_lsq.py all [N_GROUPS]   # 两者都跑
"""
import os, sys, time, pickle, warnings, math
warnings.filterwarnings('ignore')

_HERE = os.path.dirname(os.path.abspath(__file__))   # .../PSSE_Simulation/scripts
ROOT = os.path.dirname(_HERE)                          # .../PSSE_Simulation (跨机器自动适配, 无关盘符)
os.chdir(ROOT)
sys.path.append(os.path.join(ROOT, 'package'))

import numpy as np
from scipy.optimize import least_squares
from identification import _core_con, _core_con_jac   # 纯数值核(不碰 license)

OUT = os.path.join(ROOT, 'datalen_output')
os.makedirs(OUT, exist_ok=True)
PKL = os.path.join(OUT, 'datalen_data.pkl')

dt = 0.01
LB4 = [100, 30, 0.08, 0.2]     # [a(缩放后), b, H2(缩放后), P_im]
UB4 = [1200, 240, 0.22, 0.8]
PNAMES = ['a', 'b', 'H2', 'P_im']
LENS = [(10, 1001), (30, 3001), (60, 6001), (90, 9001), (120, 12001)]   # (秒, 点数)
N_IDEN = 5                      # 每条曲线每档长度辨识 5 次随机初值取最优
SEED = 20260925
PENALTY = 1e6                   # 数值发散/稳态越界时的逐点惩罚


# ==================== 采样 con 真值(与 psse_con_scan.py 口径一致) ====================
def sample_con(rng):
    """采样 con 真值。P_im ∈ [0.3, 0.7](纯 con 无 var)。"""
    while True:
        P_ma = rng.uniform(0.3, 0.7)
        X_a = rng.uniform(1.35, 2.05); X1_a = rng.uniform(0.12, 0.22)
        T0_a = rng.uniform(0.08, 0.2);  H2_a = rng.uniform(0.1, 0.2)
        pr = [(X_a / X1_a - 1) / T0_a / X1_a, X_a / X1_a / T0_a, H2_a, P_ma]
        if (np.array(pr[:3]) > np.array(LB4[:3])).all() and (np.array(pr[:3]) < np.array(UB4[:3])).all():
            cc = {'P_ma': P_ma, 'P_mb': 0.0, 'P_mc': 0.0, 'P_md': 0.0, 'P_e': 0.0,
                  'R_a': 0.0, 'X_a': X_a, 'X1_a': X1_a, 'T0_a': T0_a, 'H2_a': H2_a}
            return np.array(pr), cc


# ==================== least_squares 残差与 Jacobian ====================
def trim_k(n):
    """头尾各裁点数的比例: 1000 点裁 50(即各 5%), k = n // 20, 至少 1。

    张老师 2026-09-26 口径: k 随数据长度缩放(而非固定 100 点), 以适配不同时长仿真:
      10s(1001 点)→50, 30s→150, 60s→300, 120s(12001 点)→600。
    """
    return max(1, n // 20)

def _proj_resid(u, P, Q, P_p, Q_p):
    """对 (P_p-P, Q_p-Q) 在 [1,V,V^2] 上回归, 返回投影残差 (m,2)。

    头尾各裁 k=trim_k(n) 点: 有效区间 [k : n-k]。
    头 k 点 = 启动/滤波暂态, 末 k 点 = 仿真末端截断瞬态(单向漂移), 一并剔除。
    """
    n = len(u)
    k = trim_k(n)
    sl = slice(k, n - k)
    V = abs(u[sl])
    m = len(V)
    X = np.column_stack((np.ones(m), V, V**2))
    yP = P_p[sl] - P[sl]
    yQ = Q_p[sl] - Q[sl]
    beta = np.linalg.solve(X.T @ X, X.T @ np.column_stack((yP, yQ)))
    return np.column_stack((yP, yQ)) - X @ beta


def residuals_lsq(para, u, P, Q):
    """逐点残差向量 r = concat(resid_P, resid_Q), sum(r^2)=SS(等价 obj_con_iden)。"""
    a, b, H2, mva = para[0]/0.75, para[1], para[2]/0.75, P[0]*para[3]
    ok, P_p, Q_p = _core_con(u, a, b, H2, mva, dt)
    n = len(u)
    k = trim_k(n)
    if not ok:
        return np.full(2*(n - 2*k), PENALTY)
    resid = _proj_resid(u, P, Q, P_p, Q_p)
    return np.concatenate([resid[:, 0], resid[:, 1]])


def jac_lsq(para, u, P, Q):
    """残差向量对 4 参数(a,b,H2,P_im)的解析 Jacobian (2m,4)。"""
    a, b, H2, mva = para[0]/0.75, para[1], para[2]/0.75, P[0]*para[3]
    ok, P_p, Q_p, dPa, dQa = _core_con_jac(u, a, b, H2, mva, dt)
    n = len(u)
    k = trim_k(n)
    m = n - 2*k
    if not ok:
        return np.zeros((2*m, 4))
    sl = slice(k, n - k)
    V = abs(u[sl])
    X = np.column_stack((np.ones(m), V, V**2))
    Qr = np.linalg.solve(X.T @ X, X.T)          # (3,m), (I-H)=I-X Qr
    dY_P = np.empty((n, 4)); dY_Q = np.empty((n, 4))
    dY_P[:, 0] = dPa[:, 0] / 0.75
    dY_P[:, 1] = dPa[:, 1]
    dY_P[:, 2] = dPa[:, 2] / 0.75
    dY_P[:, 3] = P[0] * (P_p / mva)
    dY_Q[:, 0] = dQa[:, 0] / 0.75
    dY_Q[:, 1] = dQa[:, 1]
    dY_Q[:, 2] = dQa[:, 2] / 0.75
    dY_Q[:, 3] = P[0] * (Q_p / mva)
    dYP = dY_P[sl, :]; dYQ = dY_Q[sl, :]
    dRP = dYP - X @ (Qr @ dYP)                   # (m,4) 投影后 P 残差导数
    dRQ = dYQ - X @ (Qr @ dYQ)
    return np.vstack([dRP, dRQ])


def cost_of(res, n):
    """把 least_squares 结果换算成 obj_con_iden 口径的标量代价(越小越好)。"""
    k = trim_k(n)
    return math.sqrt(float(np.sum(res.fun ** 2))) / (n - 2*k) * 1000


def identify_best_lsq(u, P, Q, n_restarts=N_IDEN):
    """多次随机初值跑 least_squares, 返回 (best_x, best_cost)。"""
    best_x, best_cost = None, float('inf')
    for _ in range(n_restarts):
        # 采样可行初值(核递推 ok=True)
        x0 = None
        for _try in range(50):
            cand = np.random.uniform(LB4, UB4)
            a, b, H2, mva = cand[0]/0.75, cand[1], cand[2]/0.75, P[0]*cand[3]
            ok, _, _ = _core_con(u, a, b, H2, mva, dt)
            if ok:
                x0 = cand
                break
        if x0 is None:
            continue
        try:
            res = least_squares(residuals_lsq, x0, jac=jac_lsq, args=(u, P, Q),
                                bounds=(LB4, UB4), method='trf',
                                xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=300)
            c = cost_of(res, len(u))
            if c < best_cost:
                best_cost = c
                best_x = np.asarray(res.x)
        except Exception:
            continue
    return best_x, best_cost


# ==================== 误差指标 ====================
def relerr(x, xt):
    return np.abs(np.asarray(x) - np.asarray(xt)) / np.abs(np.asarray(xt))

def relerr_signed(x, xt):
    return (np.asarray(x) - np.asarray(xt)) / np.abs(np.asarray(xt))


# ==================== 数据生成(120s x N 组) ====================
def extract(df, busid_iden=3):
    ud, uq, P, Q = [df.values[:, [i for i in range(len(df.columns)) if (key % busid_iden) in df.columns[i]]].reshape(len(df))
                    for key in ["BUS%d UD", "BUS%d UQ", "PLOD %d[", "QLOD %d["]]
    return ud + 1j * uq, P, Q


def run_sim(n_groups):
    import psse_dynamic as ps_d
    busid_iden = 3

    kwargs = {
        'total_time': 120, 'ambient_cutoff_frequency': 2, 'disturbance_type': 'ambient_a', 'disturbance_range': 500,
        'ld_con_chng_buses': [busid_iden],
        'ld_con_chng_values': [ps_d.read_bus_params(busid=busid_iden)],
        'load_bus_monitor': [busid_iden],
    }
    data = pickle.load(open(PKL, 'rb')) if os.path.exists(PKL) else {}

    print('=' * 78, flush=True)
    print('生成 120s 真实 PSSE 数据: %d 组 con 真值 x 1 曲线  %s' % (n_groups, time.strftime('%H:%M:%S')), flush=True)
    print('采样: P_im∈[0.3,0.7] 纯 con; total_time=120s → 12001 点/曲线', flush=True)

    for g in range(n_groups):
        if g in data:
            print('group %d 已存在, 跳过' % g, flush=True)
            continue
        rng = np.random.default_rng(SEED + g)   # 每组独立种子: 可复现且避免跨运行重复
        pr_truth, cc = sample_con(rng)
        for name, val in cc.items():
            if name in ps_d.para_names:
                kwargs['ld_con_chng_values'][0][ps_d.para_names.index(name)] = val
        print('group %d 真值: a=%.3f b=%.3f H2=%.4f P_im=%.4f  (仿真中...)' %
              (g, pr_truth[0], pr_truth[1], pr_truth[2], pr_truth[3]), flush=True)
        df, d_para, sim_time = ps_d.one_time_simulation(kwargs, max_retries=8)
        if df is None:
            print('  group %d 仿真连续失败, 跳过' % g, flush=True)
            continue
        u, P, Q = extract(df)
        data[g] = {'truth': pr_truth, 'cc': cc, 'curves': [(u, P, Q)]}
        pickle.dump(data, open(PKL, 'wb'))
        print('  group %d 完成: 仿真 %.1fs, %d 点, |u|∈[%.4f,%.4f]' %
              (g, sim_time, len(u), abs(u).min(), abs(u).max()), flush=True)

    print('数据生成完毕, 共 %d 组, 落盘 %s' % (len(data), PKL), flush=True)


# ==================== 四档辨识对比 ====================
def diagnose(u, P, Q):
    """机电振荡主频/周期(用于解释时长影响的物理含义)。"""
    n = len(u)
    t = np.arange(n) * dt
    A = np.column_stack((np.ones(n), t))
    coef, *_ = np.linalg.lstsq(A, P, rcond=None)
    Pc = P - A @ coef
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(Pc * win))
    freqs = np.fft.rfftfreq(n, d=dt)
    band = (freqs >= 0.05) & (freqs <= 3.0)
    if band.any() and spec[band].max() > 0:
        f_peak = freqs[band][np.argmax(spec[band])]
        return 1.0 / f_peak
    return float('nan')


def run_iden():
    data = pickle.load(open(PKL, 'rb'))
    gids = sorted(data.keys())
    np.random.seed(SEED)

    lines = []
    lines.append('=' * 88)
    lines.append('数据长度对辨识精度的影响 (least_squares 口径)  ——  %d 组 con 真值 x 1 曲线' % len(gids))
    lines.append('五档: 10s / 30s / 60s / 90s / 120s   (同一 120s 曲线截取前 N 秒)')
    lines.append('核心判据 = P_im 与 H2 辨识误差(%%); a/b 记带符号偏差(正=偏大)')
    lines.append('=' * 88)

    errs = {L: [] for L, _ in LENS}
    signed = {L: [] for L, _ in LENS}
    core = {L: [] for L, _ in LENS}
    T_osc = {}

    for g in gids:
        rec = data[g]
        truth = rec['truth']
        u, P, Q = rec['curves'][0]
        T_osc[g] = diagnose(u, P, Q)
        for L, npts in LENS:
            us, Ps, Qs = u[:npts], P[:npts], Q[:npts]
            t0 = time.time()
            x, cost = identify_best_lsq(us, Ps, Qs)
            dt_used = time.time() - t0
            if x is None:
                lines.append('  g%-2d %3ds: 辨识失败' % (g, L))
                continue
            re = relerr(x, truth) * 100
            sg = relerr_signed(x, truth) * 100
            core_cur = float(re[[2, 3]].mean())
            errs[L].append(re)
            signed[L].append(sg)
            core[L].append(core_cur)
            lines.append('  g%-2d %3ds: 核心(P_im/H2)=%5.1f%%  a=%+6.1f%% b=%+6.1f%% H2=%+6.1f%% P_im=%+6.1f%%  (%.1fs)' %
                         (g, L, core_cur, sg[0], sg[1], sg[2], sg[3], dt_used))
        lines.append('  g%-2d 机电振荡周期 T≈%.1fs' % (g, T_osc[g]))
        lines.append('')

    lines.append('=' * 88)
    lines.append('汇总: 逐参数误差与偏差 (跨 %d 组)' % len(gids))
    lines.append('=' * 88)
    lines.append('%-6s %-10s %-9s %-9s %-9s %-9s' % ('长度', '指标', 'a', 'b', 'H2', 'P_im'))
    for L, npts in LENS:
        E = np.array(errs[L]) if errs[L] else np.zeros((0, 4))
        S = np.array(signed[L]) if signed[L] else np.zeros((0, 4))
        C = np.array(core[L]) if core[L] else np.array([])
        if E.size:
            lines.append('%-6s %-10s %-9s %-9s %-9s %-9s' %
                         ('%ds' % L, '误差中位%%', *[round(float(np.median(E[:, i])), 1) for i in range(4)]))
            lines.append('%-6s %-10s %-9s %-9s %-9s %-9s' %
                         ('%ds' % L, '误差均值%%', *[round(float(E[:, i].mean()), 1) for i in range(4)]))
            lines.append('%-6s %-10s %-9s %-9s %-9s %-9s' %
                         ('%ds' % L, '偏差符号%%', *[round(float(S[:, i].mean()), 1) for i in range(4)]))
            lines.append('  %-2ds 核心(P_im/H2)误差: 中位=%.1f%% 均值=%.1f%%' %
                         (L, float(np.median(C)), float(C.mean())))
        else:
            lines.append('%-6s (无有效结果)' % ('%ds' % L))
        lines.append('')

    lines.append('=' * 88)
    lines.append('核心结论: P_im/H2 误差随数据长度如何变化(中位 %%)')
    lines.append('=' * 88)
    for L, npts in LENS:
        C = np.array(core[L]) if core[L] else np.array([])
        if C.size:
            lines.append('  %3ds: 中位=%.2f%%  均值=%.2f%%  (n=%d)' % (L, float(np.median(C)), float(C.mean()), len(C)))

    txt = '\n'.join(lines)
    open(os.path.join(OUT, 'datalen_result.txt'), 'w', encoding='utf-8').write(txt)
    print(txt, flush=True)


# ==================== 直接短时仿真 vs 120s截取 对照 ====================
DIRECT_LENS = [5, 10, 20, 30]     # 直接仿真的时长(秒), 验证"特别短时辨识不准"(张老师 2026-09-26)
DIRECT_PKL = os.path.join(OUT, 'datalen_direct.pkl')


def run_direct_sim():
    """为每组真值额外跑一段『直接 N 秒』仿真(而非从 120s 截前 N 秒)。

    关键区别: 直接跑 N 秒, 仿真在 t=N 戛然而止, 末端自带截断瞬态;
    而从 120s 截前 N 秒, t=N 处是中段平滑信号, 无截断瞬态。
    """
    import psse_dynamic as ps_d
    os.environ['PSSE_KEEP_OUTFILE'] = '1'  # 跳过 .out 删除, 避免 safe-delete 批量确认挂起后台
    busid_iden = 3
    base = pickle.load(open(PKL, 'rb'))
    direct = pickle.load(open(DIRECT_PKL, 'rb')) if os.path.exists(DIRECT_PKL) else {}

    base_kw = {'ambient_cutoff_frequency': 2, 'disturbance_type': 'ambient_a', 'disturbance_range': 500,
               'ld_con_chng_buses': [busid_iden], 'load_bus_monitor': [busid_iden]}

    print('=' * 78, flush=True)
    print('生成直接短时仿真: %d 组 x %s 秒  %s' % (len(base), DIRECT_LENS, time.strftime('%H:%M:%S')), flush=True)

    for g in sorted(base.keys()):
        cc = base[g]['cc']
        for L in DIRECT_LENS:
            key = (int(g), int(L))
            if key in direct:
                print('  g%-2d %2ds 已存在, 跳过' % (g, L), flush=True)
                continue
            kw = dict(base_kw)
            kw['total_time'] = L
            kw['ld_con_chng_values'] = [ps_d.read_bus_params(busid=busid_iden)]
            for name, val in cc.items():
                if name in ps_d.para_names:
                    kw['ld_con_chng_values'][0][ps_d.para_names.index(name)] = val
            print('  g%-2d %2ds 直接仿真中...  %s' % (g, L, time.strftime('%H:%M:%S')), flush=True)
            df, d_para, sim_time = ps_d.one_time_simulation(kw, max_retries=8)
            if df is None:
                print('    g%-2d %2ds 仿真失败, 跳过' % (g, L), flush=True)
                continue
            u, P, Q = extract(df)
            direct[key] = {'truth': base[g]['truth'], 'u': u, 'P': P, 'Q': Q}
            pickle.dump(direct, open(DIRECT_PKL, 'wb'))
            print('    g%-2d %2ds 完成: %.1fs, %d 点' % (g, L, sim_time, len(u)), flush=True)
    print('直接仿真完毕, 共 %d 条, 落盘 %s' % (len(direct), DIRECT_PKL), flush=True)


def run_direct_iden():
    """对照『直接 N 秒』与『120s 截前 N 秒』的辨识精度, 验证短时启动/截断瞬态影响。"""
    base = pickle.load(open(PKL, 'rb'))
    direct = pickle.load(open(DIRECT_PKL, 'rb'))
    gids = sorted(base.keys())
    np.random.seed(SEED)
    lens = DIRECT_LENS

    # 预计算 {(g, L, 'cut'|'direct'): 核心(P_im/H2)误差%}
    res = {}
    for g in gids:
        truth = base[g]['truth']
        u120, P120, Q120 = base[g]['curves'][0]
        for L in lens:
            npts = L * 100 + 1
            xc, _ = identify_best_lsq(u120[:npts], P120[:npts], Q120[:npts])
            res[(g, L, 'cut')] = float(relerr(xc, truth)[[2, 3]].mean() * 100) if xc is not None else float('nan')
            key = (int(g), int(L))
            if key in direct:
                ud, Pd, Qd = direct[key]['u'], direct[key]['P'], direct[key]['Q']
                xd, _ = identify_best_lsq(ud, Pd, Qd)
                res[(g, L, 'direct')] = float(relerr(xd, truth)[[2, 3]].mean() * 100) if xd is not None else float('nan')

    lines = []
    lines.append('=' * 88)
    lines.append('直接短时仿真 vs 120s截取 的辨识精度对照 (least_squares + 头尾各裁5%%)')
    lines.append('cut    = 从同一组 120s 仿真截取前 N 秒(无末端截断瞬态)')
    lines.append('direct = 单独跑 N 秒仿真(自带末端截断瞬态)')
    lines.append('核心判据 = P_im/H2 辨识误差(%%); 越大越不准')
    lines.append('=' * 88)

    for g in gids:
        seg = []
        for L in lens:
            c = res.get((g, L, 'cut'), float('nan'))
            d = res.get((g, L, 'direct'), float('nan'))
            seg.append('  %2ds: cut=%5.1f%% direct=%5.1f%%' % (L, c, d))
        lines.append('  g%-2d:' % g + ''.join(seg))

    lines.append('')
    lines.append('-' * 88)
    lines.append('%-6s %-20s %-20s %-12s' % ('长度', 'cut核心误差中位', 'direct核心误差中位', 'direct-cut'))
    for L in lens:
        cv = [res[(g, L, 'cut')] for g in gids if (g, L, 'cut') in res and not math.isnan(res[(g, L, 'cut')])]
        dv = [res[(g, L, 'direct')] for g in gids if (g, L, 'direct') in res and not math.isnan(res[(g, L, 'direct')])]
        mc = float(np.median(cv)) if cv else float('nan')
        md = float(np.median(dv)) if dv else float('nan')
        lines.append('%-6s %-20s %-20s %-12s' %
                     ('%ds' % L,
                      '%.2f%% (n=%d)' % (mc, len(cv)),
                      '%.2f%% (n=%d)' % (md, len(dv)),
                      '%+.2f%%' % (md - mc)))

    txt = '\n'.join(lines)
    open(os.path.join(OUT, 'datalen_direct_result.txt'), 'w', encoding='utf-8').write(txt)
    print(txt, flush=True)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    n_groups = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    if mode == 'sim':
        run_sim(n_groups)
    elif mode == 'iden':
        run_iden()
    elif mode == 'all':
        run_sim(n_groups); run_iden()
    elif mode == 'direct_sim':
        run_direct_sim()
    elif mode == 'direct_iden':
        run_direct_iden()
    elif mode == 'direct':
        run_direct_sim(); run_direct_iden()
    else:
        print('未知 mode: %s (可用 sim/iden/all/direct_sim/direct_iden/direct)' % mode)
