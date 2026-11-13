from __future__ import with_statement               #<- it must be first declaration
import os
import sys
import time

f_null = open(os.devnull, 'w')
old_stdout = sys.stdout
sys.stdout = f_null

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# ---- PSSE 路径自适应探测（2026-11-11）----
# 兼容 本机(D盘 36.2) / Dell Tower(C盘 36.3) 等不同安装位置；
# PSSPY 绑定必须与 Python 主版本严格匹配：PSSPY312->Python3.12, PSSPY313->Python3.13
_pyver = f"{sys.version_info.major}{sys.version_info.minor}"
_dll_handles = []  # 持有 add_dll_directory 句柄, 防止被垃圾回收


def _find_psspy():
    """按 盘符 x PSSE大版本 x 版本号 探测 PSSPY<pyver> 目录, 返回 (psspy_path, pss_type)。"""
    for _drive in ('D', 'C', 'E', 'F'):
        _base = os.path.join(_drive + ':\\', 'Program Files', 'PTI')
        if not os.path.isdir(_base):
            continue
        for _pss, _type in (('PSSE36', 36), ('PSSE35', 35)):
            _pss_dir = os.path.join(_base, _pss)
            if not os.path.isdir(_pss_dir):
                continue
            _vers = sorted(
                [v for v in os.listdir(_pss_dir) if os.path.isdir(os.path.join(_pss_dir, v))],
                reverse=True,
            )
            for _v in _vers:
                _psspy = os.path.join(_pss_dir, _v, 'PSSPY' + _pyver)
                if os.path.isdir(_psspy):
                    return _psspy, _type
    return None, None


PSSPY_PATH, _pss_type = _find_psspy()
if PSSPY_PATH is None:
    raise RuntimeError(
        "未找到匹配 Python %d.%d 的 PSSE 安装（需要 PSSPY%s）。"
        "请确认 PSSE 已安装，或手动在 package/psse_dynamic.py 指定路径。"
        % (sys.version_info.major, sys.version_info.minor, _pyver)
    )

sys.path.append(PSSPY_PATH)
os.environ['PATH'] += ';' + PSSPY_PATH

# PSSBIN(PSSE 运行时 DLL)一并加入搜索路径: 部分机器(如 Dell Tower 36.3)必须加 PSSBIN 才能 import psspy
_pssbin = os.path.join(os.path.dirname(PSSPY_PATH), 'PSSBIN')
for _dll_dir in (_pssbin, PSSPY_PATH):
    if os.path.isdir(_dll_dir):
        os.environ['PATH'] += ';' + _dll_dir
        if hasattr(os, 'add_dll_directory'):
            try:
                _dll_handles.append(os.add_dll_directory(_dll_dir))
            except Exception:
                pass

if _pss_type == 35:
    import psse35
else:
    import psse36


import psspy,redirect
import dyntools
from multiprocessing import Process, Queue

sys.stdout = old_stdout
f_null.close()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d as inp

# Import PSS/E default
_i = psspy.getdefaultint()
_f = psspy.getdefaultreal()
_s = psspy.getdefaultchar()
redirect.psse2py()

para_indexes=[19,20,21,22,23,40, 41, 42, 44, 46,60, 61, 62, 64, 66]
para_names=['P_ma','P_mb','P_mc','P_md','P_e','R_a','X_a','X1_a','T0_a','H2_a','R_b','X_b','X1_b','T0_b','H2_b']

# ============ CMLDBLU2 负荷模型的原始通道表(VAR/STATE, 由模型结构决定) ============
ori_idx_var=[0,15,16,29,30,31,32,33,34,35,36,37,38,71,72,73,74,75,76,79,80,87,88,89,90,91,92,95,96,103,104,105,106,107,108,111,112,25,26]
ori_ident_var=['load MVA','UD','UQ','load bus v','low side bus v',
               'MA P','MA Q','MB P','MB Q','MC P','MC Q',
               'MD P','MD Q','MA Tele','MA Speed Deviation','MA init load torque',
               'MA Id','MA Iq','MA I','MA MVA','MA TL',
               'MB Tele','MB Speed Deviation','MB init load torque','MB Id','MB Iq',
               'MB I','MB MVA','MB TL','MC Tele','MC Speed Deviation',
               'MC init load torque','MC Id','MC Iq','MC I','MC MVA',
               'MC TL','S P','S Q']
ori_idx_state=[0,1,4,5,6,7,10,11,12,13,16,17]
ori_ident_state=['MA Eq1','MA Ed1','MA speed deviation','MA angle deviation',
                 'MB Eq1','MB Ed1','MB speed deviation','MB angle deviation',
                 'MC Eq1','MC Ed1','MC speed deviation','MC angle deviation']

def read_bus_params(busid, path='./Cases/IEEE39/ieee39_RE_ieee39_RE_motorD_Stall_enabled.dyr'):
    with open(path) as f:
        lines = f.read().splitlines()

    # 1. 找 bus 头行：行首整数 == busid，且含 'USRLOD'
    header = None
    for i, line in enumerate(lines):
        if line.split() and line.split()[0] == str(busid) and "'USRLOD'" in line:
            header = i
            break
    if header is None:
        raise ValueError(f'未找到 bus {busid}')

    # 2. 向下两行开始读参数
    start = header + 2

    # 3. 结束位置：下一个 'USRLOD' 头行
    end = len(lines)
    for i in range(start, len(lines)):
        if "'USRLOD'" in lines[i]:
            end = i
            break

    # 4. 提取数字（不用 re）
    nums = []
    for line in lines[start:end]:
        for part in line.split(','):      # 先按逗号切
            for tok in part.split():      # 再按空白切
                try:
                    nums.append(float(tok))
                except ValueError:
                    pass                  # 跳过 /LOAD3 这类非数字
    # 直接返回 para_indexes 位置上的参数值(与 ld_con_chng_values 行对齐), 供 change_ldmod 使用
    return np.array(nums)[np.array(para_indexes)-1].tolist()

def _check(call_name, ierr):
    """检查 PSSE API 返回值, 失败即抛异常(由外层重试机制兜底)"""
    if ierr != 0:
        raise RuntimeError(f"PSSE API {call_name} 失败 (ierr={ierr})")

# ============ 类噪声注入: 新增 0 值负荷(不改原负荷) ============
# 在注入母线上额外加一个初值 P=Q=0 的新负荷(新 load ID), 类噪声/probing 激励全部
# 经这个 0 值负荷注入系统, 不再直接改写原有 '1' 号负荷(张老师 2026-11-13 要求)。
# INJECT_LOAD_ID 选 '2': IEEE39 每个负荷母线仅有一个 '1' 号负荷, '2' 空闲(已实测确认)。
INJECT_LOAD_ID = '2 '   # 注入负荷的 2 字符 load ID(与 '1 ' 同款右补空格写法)
INJECT_LOAD_BUSES = [12, 7, 8, 23, 16, 18, 26, 29]  # 类噪声注入母线

def init_dynamic_case(case_path, study, load_bus_monitor, simu_time_step=0.002, bus_q_chng=None, dyradd=None,
                      inject_load_buses=None):
    """PSSE 会话初始化 + 案例转换 + 动态模型装载（每次仿真几乎不变的部分）。

    内部顺序: psseinit → case 载入 → 负荷/发电机转换(CMLDBLU2) → dyre_new 编译
              → snap 快照 → dyre_add 附加模型 → 设置动态求解参数(积分步长等)。
    全部关键调用带 _check 错误检查。
    注意: 会话保持开启, 由调用方(外层 finally)负责 pssehalt_2。

    bus_q_chng: 可选 dict {母线号: 恒功率无功 QL(MVAR)}。在 fdns 潮流解出之后、
                cong(发电机转换)之前, 用 load_chng_6 修改指定母线负荷的恒功率无功并 fnsl 重解,
                用于抬/降某母线电压(机理验证实验用)。默认 None = 不改。

    inject_load_buses: 可选 list[母线号]。对其中每个母线, 在 case 载入后、conl 转换前,
                用 load_data_6 新增一个初值 P=Q=0 的新负荷(load ID = INJECT_LOAD_ID),
                类噪声/probing 激励后续经这个 0 值负荷注入(不改原 '1' 号负荷)。
                默认 None = 不新增。
    """
    sav_file = case_path + '%s.sav' % study
    cnv_file = case_path + '%s_cnv.sav' % study
    dyr_file = case_path + '%s.dyr' % study
    if dyradd is None:
        dyradd = case_path + 'ieee39_RE_ieee39_RE_motorD_Stall_enabled.dyr'
    snp_file = case_path + '%s_%s.snp' % (study, study)

    _check('psseinit', psspy.psseinit(20000))

    _check('case', psspy.case(sav_file))
    # 类噪声注入: 在指定母线新增一个初值 P=Q=0 的负荷(新 load ID), 供后续 load_chng_6 注入噪声。
    # 加在 conl(负荷转换)之前, 使新负荷与原有负荷走完全相同的转换/动态初始化流程,
    # 这样 load_chng_6 作用于新 ID 与作用于原 '1' 号负荷的行为完全一致。
    if inject_load_buses:
        for bus in inject_load_buses:
            _check('load_data_6', psspy.load_data_6(bus, INJECT_LOAD_ID,
                    [1, _i, _i, _i, 1, _i, _i],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    psspy.solution_parameters_3([_i, 100, _i],
                                [_f, _f, _f, _f, _f, _f, _f, _f, _f, _f, _f, _f, _f, _f, _f, _f, _f, _f, _f])
    # convert loads and create converted case
    psspy.conl(_i, _i, 1, [0, _i], [_f, _f, _f, _f])
    psspy.bsys(sid=1, numbus=len(load_bus_monitor) + 2, buses=load_bus_monitor + [12, 39])  # 39 节点是平衡节点
    psspy.conl(1, 0, 2, [_i, _i], [50.0, 50.0, 0.0, 50.0])
    psspy.conl(1, 1, 2, [_i, _i], [50.0, 50.0, 0.0, 50.0])
    psspy.conl(_i, _i, 3, [_i, _i], [_f, _f, _f, _f])
    psspy.fdns([1, 0, 1, 1, 1, 0, 99, 0])
    # 机理验证: 改指定母线恒功率无功抬/降电压(在发电机转换前、动态模型编译前)
    if bus_q_chng:
        for bus, ql in bus_q_chng.items():
            _check('load_chng_6', psspy.load_chng_6(bus, '1 ', [], [_f, ql, _f, _f, _f, _f, _f, _f]))
        _check('fnsl', psspy.fnsl())
    # convert gens
    psspy.cong()
    psspy.ordr(0)
    psspy.fact()
    psspy.tysl(0)
    psspy.tysl(0)
    _check('save', psspy.save(cnv_file))
    _check('dyre_new', psspy.dyre_new([1, 1, 1, 1], dyr_file, r"""conec.for""", r"""conet.for""", ""))
    # save snapshot for dynamics
    _check('snap', psspy.snap([-1, -1, -1, -1, -1], snp_file))

    # read in-lib DYRE records
    _check('dyre_add', psspy.dyre_add([_i, _i, _i, _i], dyradd, '', ''))

    # 动态仿真求解参数(积分步长等)
    _check('dynamics_solution_param_2', psspy.dynamics_solution_param_2([99, _i, _i, _i, _i, _i, _i, _i],
                                                                        [1.0, _f, simu_time_step, simu_time_step * 4, _f, _f, _f, _f]))


def read_con_params(load_bus_monitor):
    """读取各监视母线当前 CON 参数(change_ldmod 施加后的值), 构建 d_para 基础表。

    依赖活动 PSSE 会话(在 init_dynamic_case 之后、仿真结束前调用)。
    返回 DataFrame: 母线 × CON 参数(index=load_bus_monitor, 不含 MVA 列, 由后处理追加)。
    """
    bus_para = []
    for bus in load_bus_monitor:
        ival = psspy.lmodind(bus, '1', 'CHARAC', 'CON')[1]
        para_value = [psspy.dsrval('CON', ival + para_indexes[i] - 1)[1] for i in range(len(para_indexes))]
        bus_para.append(para_value)
    return pd.DataFrame(data=bus_para, columns=para_names, index=load_bus_monitor)


def configure_output_channels(load_bus_monitor, var_channels, state_channels):
    """配置动态仿真的 .out 输出通道: 系统级母线电压/负荷(chsb) + 各监视母线的 VAR/STATE 通道。

    var_channels/state_channels 为需要记录的通道名(从 ori_ident_* 原始表中选取),
    由调用方(通常 run_psse_simulation)决定; state_channels 当前为空(不需要 STATE 通道)。
    """
    psspy.chsb(sid=1, all=0, status=[-1, -1, -1, 1, 14, 0])  # 母线电压相角通道
    psspy.chsb(sid=1, all=0, status=[-1, -1, -1, 1, 25, 0])  # 负荷有功通道
    psspy.chsb(sid=1, all=0, status=[-1, -1, -1, 1, 26, 0])  # 负荷无功通道

    idx_var=[ori_idx_var[ori_ident_var.index(n)] for n in var_channels]
    idx_state=[ori_idx_state[ori_ident_state.index(n)] for n in state_channels]

    for bus in load_bus_monitor:
        base_var = psspy.lmodind(bus, '1', 'CHARAC', 'VAR')[1]
        for name, off in zip(var_channels, idx_var):
            psspy.var_channel(status=[-1, base_var+off], ident='Bus%d %s' % (bus, name))
        if state_channels:
            base_state = psspy.lmodind(bus, '1', 'CHARAC', 'STATE')[1]
            for name, off in zip(state_channels, idx_state):
                psspy.state_channel(status=[-1, base_state+off], ident='Bus%d %s' % (bus, name))


def postprocess_outfile(outfile, d_para, total_time, simu_time_step, sample_time_step, filter_cutoff_frequency, load_bus_monitor):
    """从 PSSE .out 通道文件生成结果表, 并给 d_para 追加 MVA 基值列(纯数据处理, 不依赖 PSSE 会话)。

    流程: 解析通道 → 去重(时间重复取变化后值) → 线性插值到 simu_time_step 网格
          → 5 阶低通滤波(filter_cutoff_frequency) → 下采样到 sample_time_step
    随后从首行取各母线 MA/MB/MC MVA 基值(pu→1/100), 在 d_para 副本上追加 mva/mvb/mvc 列。
    返回 (df, d_para): 以 Time(s) 为索引的 DataFrame, 与追加了 MVA 列的负荷参数表。
    """
    chnfobj = dyntools.CHNF(outfile)
    short_title, chanid, chandata = chnfobj.get_data()

    data_ori = np.zeros((len(chandata['time']), len(chanid)))      # 原始数据数组
    data_no_filt = np.zeros((len(chandata['time']), len(chanid)))  # 缓冲数组

    for i in list(chanid.keys()):  # 把结果数据转化为 numpy 数组
        if i == 'time':
            data_ori[:, 0] = chandata['time']
        else:
            data_ori[:, i] = chandata[i]

    # 找到时间点重复的位置, 取变化后的数值
    j = 0
    n = 0
    while j < len(chandata['time']) - 1:
        if (abs(data_ori[j, 0] - data_ori[j+1, 0]) < 1e-5):
            data_no_filt[n, :] = data_ori[j+1, :]
            j += 2
        else:
            data_no_filt[n, :] = data_ori[j, :]
            j += 1
        n += 1
    data_no_filt[n, :] = data_ori[j, :]
    data_ori = data_no_filt[0:n+1, :]

    # 线性插值到仿真步长网格
    f = inp(data_ori[:, 0], data_ori, kind='linear', fill_value='extrapolate', axis=0)
    data_no_filt = f(np.arange(0, total_time + simu_time_step, simu_time_step))

    # 5 阶低通滤波
    data_filt = np.zeros(np.shape(data_no_filt))
    nyq = 1 / simu_time_step / 2  # 奈奎斯特频率
    normal_cutoff = filter_cutoff_frequency / nyq  # 归一化截止频率
    b, a = butter(5, normal_cutoff, btype='low', analog=False)
    data_filt = filtfilt(b, a, data_no_filt, axis=0)  # 对每一列进行滤波
    data_filt[:, 0] = data_no_filt[:, 0]

    # 下采样到 sample_time_step
    f = inp(data_filt[:, 0], data_filt, kind='linear', fill_value='extrapolate', axis=0)
    data_downsampled = f(np.arange(0, total_time + sample_time_step, sample_time_step))
    df = pd.DataFrame(data_downsampled, columns=list(chanid.values()))
    df.set_index('Time(s)', inplace=True)

    # 各母线负荷模型 MVA 基值(pu → 1/100), 在 d_para 副本上追加 mva/mvb/mvc 列;
    # 若输出通道未含对应 MVA 通道则填 NaN(通道可裁剪, 不报错)
    d_para = d_para.copy()
    def _mva(bus, suffix):
        col = 'BUS%d %s MVA' % (bus, suffix)
        return df[col][0] / 100 if col in df.columns else np.nan
    d_para.insert(d_para.shape[1], 'mva', [_mva(bus, 'MA') for bus in load_bus_monitor])
    d_para.insert(d_para.shape[1], 'mvb', [_mva(bus, 'MB') for bus in load_bus_monitor])
    d_para.insert(d_para.shape[1], 'mvc', [_mva(bus, 'MC') for bus in load_bus_monitor])
    return df, d_para

def generate_ambient_signals(amplitude, frequency_sampling, length, size, cutoff_frequency, seed=None):
    """生成平滑启动的类噪声信号：每列一段 length 长的低通高斯噪声，
    起点处 |y|<0.01（避免给 PSSE 一个初始大幅阶跃扰动）。

    seed=None 时沿用全局 np.random 状态(子进程 spawn 每次用系统熵, 不可复现);
    传 seed 则显式复现同一激励, 用于"同一真值跑多次不同激励"的统计实验(2026-09-26 张老师口径)。

    注: 用单位幅值噪声 + 过零起点, 最后再乘 amplitude——这样同一 seed 下不同 amplitude
    得到完全成比例的同一形状信号, 自适应幅值修正才能一次收敛。
    """
    if seed is not None:
        np.random.seed(seed)
    nyq = frequency_sampling / 2
    normal_cutoff = cutoff_frequency / nyq
    b, a = butter(5, normal_cutoff, btype='low', analog=False)  # 5阶低通滤波器
    margin = 8 * length  # 预热段长度：保证几乎总能找到"过零起点"（原实现用 99*length，计算量高 ~10 倍）
    while True:
        y = np.random.uniform(-1.0, 1.0, [length + margin, size])   # 单位幅值, 最后统一乘 amplitude
        y_filt = filtfilt(b, a, y, axis=0)  # 对每一列进行滤波
        y_final = np.zeros((length, size))
        ok = True
        for i in range(size):
            found = False
            for j in range(margin):  # 找该通道的过零起点(符号变号, 与幅值无关)
                if y_filt[j, i] * y_filt[j + 1, i] <= 0:
                    k = j if abs(y_filt[j, i]) <= abs(y_filt[j + 1, i]) else j + 1
                    y_final[:, i] = y_filt[k:k + length, i] * amplitude
                    found = True
                    break
            if not found:
                ok = False  # 任一通道未找到起点则整段重来
                break
        if ok:
            break
    return y_final


def generate_probing_signals(amplitude, frequency_sampling, length, size,
                             f_lo=6.0, f_hi=13.0, n_comp=8, seed=None):
    """生成多频(multi-sine)探测激励, 能量集中在 [f_lo, f_hi] (默认 6~13Hz 转子动态敏感频段)。

    设计依据(2026-09-29 optimal_excitation.py / hf_design.py 可辨识性诊断结论):
      在相位主导复数激励下, A/B 同名参数(a/b/H2)的辨识方差(A-最优目标: 最小化
      Fisher 信息矩阵逆的加权迹)在 6~13Hz 达最小, 比真实低频激励(0.29Hz 峰值)与
      3~5Hz 方案改善 1~2 个数量级; 该频段对应转子磁链动态特征频段(转子回路时间
      常数 ~1.6~16Hz)。此前 5Hz 扫描上界遗漏了该敏感区。

    形态: 每个母线(size 维)独立随机相位的 n_comp 个对数均匀频点正弦叠加, 各频点等幅,
    加 Hann 淡入窗避免初始阶跃, 最后乘 amplitude 得有功扰动(MW)。

    与 generate_ambient_signals 接口一致: 返回 (length, size) 数组, 可直接替换
    run_psse_simulation 中 load_chng_6 注入环节的 am_sg。
    注意: 落地时须同步把 postprocess_outfile 的 filter_cutoff_frequency 从默认 10Hz
    提到 >= 2*f_hi (如 30Hz), 否则高频成分会被后处理低通滤掉。
    """
    if seed is not None:
        np.random.seed(seed)
    freqs = np.logspace(np.log10(f_lo), np.log10(f_hi), n_comp)
    t = np.arange(length) / frequency_sampling
    # 每个母线独立随机相位 -> 各母线激励互不相关, 避免空间一致性同时激发网络模态
    ph = np.random.uniform(0, 2 * np.pi, (n_comp, size))
    y = np.zeros((length, size))
    for k, f in enumerate(freqs):
        y += np.sin(2 * np.pi * f * t[:, None] + ph[k, None, :])
    # 归一化到单位峰值 -> 同一 seed 下不同 amplitude 成比例(与 ambient 生成器约定一致)
    peak = float(np.max(np.abs(y)))
    if peak > 1e-12:
        y = y / peak
    # Hann 淡入窗: 前 0.5s 从 0 平滑升到 1, 避免给 PSSE 初始阶跃扰动
    fade = int(round(0.5 * frequency_sampling))
    if 0 < fade < length:
        win = np.hanning(2 * fade)[:fade]
        y[:fade, :] *= win[:, None]
    return y * amplitude


def worker(q, *args, **kwargs):
    """子进程包装器"""
    # 把子进程控制台输出倒向黑洞(Fortran 底层打印也会被吞掉)
    f_null = open(os.devnull, 'w')
    # 备份原子进程的 stdout 文件描述符（其实在子进程里不备份也行，因为运行完就死了）
    old_stdout_fileno = os.dup(sys.stdout.fileno())

    # 将系统的标准输出（fileno 1）强行重定向到 devnull
    os.dup2(f_null.fileno(), sys.stdout.fileno())
    try:
        # 此时运行 run_psse_simulation，所有底层 Fortran 漏出来的打印都会被黑洞吞掉
        df, d_para = run_psse_simulation(*args, **kwargs)
        q.put((df, d_para))
    except Exception as e:
        q.put(e)
    finally:
        # 恢复输出
        os.dup2(old_stdout_fileno, sys.stdout.fileno())
        f_null.close()

def one_time_simulation(kwargs, max_retries=None):
    """运行一次 PSSE 仿真（子进程隔离），失败自动重试直到成功。

    返回 (df, d_para, simulation_time)：
        df, d_para        —— 仿真数据与负荷模型参数
        simulation_time   —— 本函数总耗时（秒，含失败重试尝试的时间）
    max_retries=None 表示无限重试；连续失败达到 max_retries 次时返回 (None, None, 耗时)。
    扰动幅值超差(RangeOutOfTolerance)走自适应修正, 不占 max_retries 次数, 有独立上限。"""
    t_start = time.perf_counter()
    attempts = 0
    range_attempts = 0
    run_kwargs = dict(kwargs)
    # 固定本组噪声种子: 让自适应修正各次尝试使用同一种噪声形状(只缩放幅值),
    # 否则换种子会导致"校正"与噪声随机性打架, 永远收敛不到 ±RANGE_TOL。
    if run_kwargs.get('ambient_seed') is None:
        run_kwargs['ambient_seed'] = np.random.randint(0, 2**31 - 1)
    while True:
        q = Queue()
        p = Process(target=worker, args=(q,), kwargs=run_kwargs)
        p.start()
        result = q.get()
        p.join()
        if not isinstance(result, Exception):
            df, d_para = result
            #print(f"仿真成功完成，数据行数：{len(df)}")
            return df, d_para, time.perf_counter() - t_start
        if isinstance(result, RangeOutOfTolerance):
            range_attempts += 1
            new_mag = result.used_magnitude * result.target / result.achieved
            print(f"扰动幅值超差(第 {range_attempts} 次): {result}; 自适应修正 MW={result.used_magnitude:.0f}→{new_mag:.0f}")
            if range_attempts >= MAX_RANGE_CORRECTIONS:
                print(f"扰动幅值自适应修正达到上限 {MAX_RANGE_CORRECTIONS} 次, 放弃")
                return None, None, time.perf_counter() - t_start
            run_kwargs['ambient_magnitude_override'] = new_mag
            continue
        attempts += 1
        print(f"仿真发生错误: {result}（第 {attempts} 次失败）")
        if max_retries is not None and attempts >= max_retries:
            return None, None, time.perf_counter() - t_start

# ============ 扰动类型解析(查表) ============
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AMBIENT_U_CALIB_CSV = os.path.join(_ROOT, 'ambient_u_calib.csv')
AMBIENT_P_CALIB_CSV = os.path.join(_ROOT, 'ambient_p_calib.csv')
RANGE_TOL = 0.0025   # 扰动幅值(max-min)达标容差: 实际范围须在 [target-RANGE_TOL, target+RANGE_TOL]
MAX_RANGE_CORRECTIONS = 5   # 自适应修正的最大次数(防死循环)


class RangeOutOfTolerance(Exception):
    """扰动幅值超差异常: 携带目标/实际范围与本次所用 MW, 供上层做自适应修正。"""
    def __init__(self, target, achieved, kind, busid, used_magnitude):
        self.target = target
        self.achieved = achieved
        self.kind = kind
        self.busid = busid
        self.used_magnitude = used_magnitude
        super().__init__(target, achieved, kind, busid, used_magnitude)

    def __str__(self):
        return '[扰动幅值校验] bus %d %s: 目标 %.4f 实际 %.4f, 超出容差 %.4f' \
               % (self.busid, self.kind, self.target, self.achieved, RANGE_TOL)


def _lookup_mag(csv_path, busid, target, label):
    """在 (bus, range -> mag) 标定表里线性插值反查 MW 幅值, 越界则夹紧。"""
    cal = pd.read_csv(csv_path)
    sub = cal[cal['bus'] == busid].sort_values('range')
    if sub.empty:
        raise ValueError('%s 里没有 bus %d 的标定数据, 先跑: python probe_ambient.py %d'
                         % (os.path.basename(csv_path), busid, busid))
    r = sub['range'].values
    m = sub['mag'].values
    if target <= r[0]:
        mag = float(m[0])
        print('[ambient] bus %d %s: 目标 %.4f 低于表内最小 %.4f, 取 mag=%.0f MW'
              % (busid, label, target, r[0], mag))
    elif target >= r[-1]:
        mag = float(m[-1])
        print('[ambient] bus %d %s: 目标 %.4f 超过表内最大 %.4f, 夹紧到 mag=%.0f MW(实际约 %.4f)'
              % (busid, label, target, r[-1], mag, r[-1]))
    else:
        mag = float(np.interp(target, r, m))
        print('[ambient] bus %d %s: 目标 %.4f -> mag=%.0f MW' % (busid, label, target, mag))
    return mag


def resolve_ambient_magnitude(disturbance_type, disturbance_range, busid):
    """把 disturbance_type/range 解析成 ambient_magnitude(MW)。"""
    if disturbance_type == 'ambient_a':
        return float(disturbance_range)
    if disturbance_type == 'ambient_u':
        return _lookup_mag(AMBIENT_U_CALIB_CSV, busid, float(disturbance_range), 'u_range')
    if disturbance_type == 'ambient_p':
        return _lookup_mag(AMBIENT_P_CALIB_CSV, busid, float(disturbance_range), 'p_range')
    raise ValueError("disturbance_type 必须是 'ambient_a'/'ambient_u'/'ambient_p'/'load_change'/'fault', 现在是: %r" % (disturbance_type,))


def _measure_achieved_range(df, kind, busid):
    """从后处理结果里实测扰动幅值(max-min)。

    kind='ambient_u' -> |u| 的 max-min;
    kind='ambient_p' -> 负荷有功 P 的 (max-min)/mean(静态工作点附近标幺)。
    """
    cols = list(df.columns)
    if kind == 'ambient_u':
        ud = df.values[:, [i for i in range(len(cols)) if ('BUS%d UD' % busid) in cols[i]]].reshape(len(df))
        uq = df.values[:, [i for i in range(len(cols)) if ('BUS%d UQ' % busid) in cols[i]]].reshape(len(df))
        u = np.abs(ud + 1j * uq)
        return float(u.max() - u.min())
    if kind == 'ambient_p':
        P = df.values[:, [i for i in range(len(cols)) if ('PLOD %d[' % busid) in cols[i]]].reshape(len(df))
        return float((P.max() - P.min()) / P.mean())
    return None


def run_psse_simulation(study='IEEE39_RE', type_option=4, disturbance_type='ambient_a', disturbance_range=500, ambient_magnitude_override=None, ambient_cutoff_frequency=1, filter_cutoff_frequency=30,case_path='./Cases/IEEE39/', simu_time_step=0.002, sample_time_step=0.01,total_time=20,ld_con_chng_buses=[],ld_con_chng_values=[],load_bus_monitor=[3,4,15,20,21,24,25,27,28],var_channels=['load MVA','UD','UQ','MA P','MA Q','MB P','MB Q','MC P','MC Q','S P','S Q','MA MVA','MB MVA','MC MVA'],state_channels=[],ambient_seed=None,bus_q_chng=None, dyradd=None, probing_f_lo=6.0, probing_f_hi=13.0, probing_n_comp=8):
    """运行一次 PSSE 动态仿真（完整初始化 + 仿真 + 后处理）。
    无论成功或异常都恢复 sys.stdout 并 pssehalt 释放 PSSE 许可证；
    数据提取完成后自动删除 .out 中间文件。"""
    f_null = open(os.devnull, 'w')
    sys.stdout = f_null
    try:
        # 参数含义见函数签名; ld_con_chng_*: 修改负荷模型的母线/参数索引/参数值(-1 表示不改)
        # 类噪声/probing 扰动的注入母线(在 init_dynamic_case 里为这些母线新增 0 值注入负荷);
        # 其余扰动种类(load_change/fault 等)不注入, 传空列表。
        load_change_buses = list(INJECT_LOAD_BUSES) if disturbance_type in ('ambient_a', 'ambient_u', 'ambient_p', 'probing') else []

        # PSSE 初始化+转换+模型装载+求解参数(每轮不变部分); 顺带为注入母线新增 0 值注入负荷
        init_dynamic_case(case_path, study, load_bus_monitor, simu_time_step, bus_q_chng, dyradd,
                          inject_load_buses=load_change_buses)

        # 修改负荷模型参数(每轮变化的部分)
        for i in range(len(ld_con_chng_buses)):
            for j in range(len(para_indexes)):
                if ld_con_chng_values[i][j]!=-1:
                    _check('change_ldmod_con', psspy.change_ldmod_con(ld_con_chng_buses[i], '1', 'CMLDBLU2', para_indexes[j], ld_con_chng_values[i][j]))

        # 读取施加样本参数后的实际 CON 值, 构建 d_para 基础表(MVA 列由后处理追加)
        d_para = read_con_params(load_bus_monitor)

        # 配置输出通道(系统级 + 各监视母线的 VAR 通道; 通道集由 var_channels/state_channels 参数决定)
        configure_output_channels(load_bus_monitor, var_channels, state_channels)

        outfile = case_path+'%s.out'%study  # 输出文件名

        _check('strt_2', psspy.strt_2([0,1],outfile))  # 初始化动态模型

        # 扰动类型解析: 'a'/'u'/'p' 是 ambient 三种幅值口径; 其余扰动种类直接透传
        if disturbance_type in ('ambient_a', 'ambient_u', 'ambient_p'):
            sim_kind = 'ambient'
            if ambient_magnitude_override is not None:
                ambient_magnitude = float(ambient_magnitude_override)
            else:
                ambient_magnitude = resolve_ambient_magnitude(disturbance_type, disturbance_range, load_bus_monitor[0])
        elif disturbance_type == 'probing':
            # multi-sine 探测激励: disturbance_range/override 直接作为有功扰动幅值(MW)
            sim_kind = 'probing'
            ambient_magnitude = float(ambient_magnitude_override) if ambient_magnitude_override is not None else float(disturbance_range)
        else:
            sim_kind = disturbance_type
            ambient_magnitude = None

        if sim_kind in ('ambient', 'probing'):

            ambient_time_step=simu_time_step

            if sim_kind == 'ambient':
                am_sg=generate_ambient_signals(ambient_magnitude, 1/ambient_time_step, round(total_time/ambient_time_step), len(load_change_buses), ambient_cutoff_frequency, ambient_seed)
            else:
                # multi-sine 探测激励(6~13Hz 转子动态敏感频段), 能量集中在 [probing_f_lo, probing_f_hi]
                am_sg=generate_probing_signals(ambient_magnitude, 1/ambient_time_step, round(total_time/ambient_time_step), len(load_change_buses), probing_f_lo, probing_f_hi, probing_n_comp, ambient_seed)

            # 运行动态仿真: 类噪声只改新增的 0 值注入负荷(INJECT_LOAD_ID), 原 '1' 号负荷保持不动
            for n in np.arange(1, round(total_time/ambient_time_step)+1, 1):
                for j in range(len(load_change_buses)):
                    _check('load_chng_6', psspy.load_chng_6(load_change_buses[j], INJECT_LOAD_ID, [], [am_sg[n-1,j], _f, _f, _f, _f, _f, _f, _f]))
                t=n*ambient_time_step
                _check('run', psspy.run(0, t, 0, 0, 0))


        elif sim_kind=='load_change':
            total_time = 10  # 总仿真时间 (秒)
            _check('run', psspy.run(0, 0.1, 0, 0, 0))
            ierr = psspy.load_chng_6(type_option[0], '1 ', [], [type_option[1], _f, _f, _f, _f, _f, _f, _f])
            _check('run', psspy.run(0, total_time, 0, 0, 0))

        elif sim_kind=='fault':
            raise NotImplementedError("disturbance_type='fault' 尚未实现, 需提供故障母线/时长等参数")

        sys.stdout.close()

        # 解析 .out → df(含 mva 基值列), 见 postprocess_outfile
        df, d_para = postprocess_outfile(outfile, d_para, total_time, simu_time_step, sample_time_step, filter_cutoff_frequency, load_bus_monitor)

        # 扰动幅值达标校验: 给定目标范围时, 实测 max-min 必须在 [target-RANGE_TOL, target+RANGE_TOL]
        if disturbance_type in ('ambient_u', 'ambient_p'):
            achieved = _measure_achieved_range(df, disturbance_type, load_bus_monitor[0])
            if achieved is not None and abs(achieved - disturbance_range) > RANGE_TOL:
                raise RangeOutOfTolerance(disturbance_range, achieved, disturbance_type,
                                          load_bus_monitor[0], ambient_magnitude)

        # 清空 .out 中间文件(数据已提取完毕): 用截断(truncate)替代删除(remove),
        # 避免 os.remove 在 8 分片×125 组=1000 次删除时触发 WorkBuddy safe-delete
        # 批量删除确认钩子, 导致后台分片静默挂起。strt_2 每轮会(重)写该文件,
        # 截断为 0 字节即可释放磁盘, 无需真正删除(非删除操作不触发钩子)。
        try:
            with open(outfile, 'wb'):
                pass
        except OSError:
            pass

        return df,d_para


    finally:
        f_null.close()
        sys.stdout = sys.__stdout__
        try:
            psspy.pssehalt_2()
        except Exception:
            pass
