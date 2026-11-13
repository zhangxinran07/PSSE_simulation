# Bus20 馈线阻抗辨识实验 — 代码交接说明

> 用途：交接给 Claude / 在 VSCode 里继续改代码用。所有路径均为本机绝对路径。
>
> ⚠️ **「本机」指当时写这份说明的那台电脑，路径不可跨机器照抄**：PSSE 每台电脑装的位置不一样——本机（D盘主电脑）在 `D:\Program Files\PTI\`（`PSSE35\35.5`、`PSSE36\36.2`），而 `package/do_simulation.py:4` 写死的 `C:\Program Files\PTI\PSSE35\35.5\PSSPY39` 是 C 盘装 PSSE 那台的写法。**换机器时先确认自己的 PTI 目录 + 与本机 Python 版本匹配的 PSSPY 子目录**。详见《研究结论.md》C.8。

---

## 一、研究目标（一句话）

确认 **Bus20 的馈线阻抗（`Rfdr = Xfdr = 0.04, Xxf = 0.08`）是导致其电机模型 a 参数辨识系统性偏小 −48% 的根本原因**；把 Bus20 馈线清零（对齐 Bus3/4，`Rfdr=Xfdr=Xxf=0`）后，跑全幅值扫描（con + var 两种电机模型），与 Bus3/4 做同口径对比。

---

## 二、文件路径清单

### 1. 核心仿真代码（项目目录，华为家庭存储网络盘 D 盘）

| 文件 | 作用 |
|---|---|
| `D:/张欣然-F7D9CE/华为家庭存储/BUAA_research/BUAA2021/PSSE_Simulation/package/psse_dynamic.py` | PSSE 36 动态仿真封装。**本次已改**：`init_dynamic_case()` / `run_psse_simulation()` 新增可选参数 `dyradd`，用于切换 dyr 副本。 |
| `D:/张欣然-F7D9CE/华为家庭存储/BUAA_research/BUAA2021/PSSE_Simulation/Cases/IEEE39/ieee39_RE_ieee39_RE_motorD_Stall_enabled_noFdr20.dyr` | **馈线清零副本**：Bus20 的 `0.04,0.04,0.75,0.08` → `0,0,0.75,0`（对齐 Bus3）。 |

### 2. 辨识与扫描脚本（✅ 已搬到项目 `scripts/` 目录，路径自动推断、跨机器可移植）

| 文件 | 作用 |
|---|---|
| `scripts/psse_datalen_lsq.py` | **con（恒转矩）辨识模块**：`sample_con` / `residuals_lsq` / `jac_lsq` / `identify_best_lsq` / `relerr` / `relerr_signed` / `trim_k` / `cost_of`。 |
| `scripts/psse_datalen_var.py` | **var（变转矩）辨识模块**：`sample_var` / `identify_best_var_lsq`，与 con 同构，复用 con 的纯函数工具。 |
| `scripts/bus20_nofdr_sweep.py` | **全幅值扫描脚本**（con + var 双模型、断点续跑、每档统计、report 模式）。 |
| `scripts/analyze_a_bus20.py` | a 参数偏差分析脚本（口径确认用）。 |

### 3. 数据文件

**本次扫描结果（`sweep_output/` 目录，NAS）：**
- `sweep_output/bus20_nofdr_sweep_con.pkl`（con，✅ 已完成 300/300 条）
- `sweep_output/bus20_nofdr_sweep_con_result.txt`（con 汇总文本）
- `sweep_output/bus20_nofdr_sweep_var.pkl`（var，待另一台电脑跑）

**历史对比数据（Bus3/4/20 旧，另一个工作区目录）：**
```
C:/Users/Lenovo/WorkBuddy/2026-09-23-17-27-51/psse_datalen/
├── mag_sweep2.pkl             # Bus3 con（13 档 × 100）
├── mag_sweep_var_bus3.pkl     # Bus3 var（10 档 × 100）
├── mag_sweep_bus4.pkl         # Bus4 con（12 档 × 100）
├── mag_sweep_var_bus4.pkl     # Bus4 var（10 档 × 100）
├── mag_sweep_bus20.pkl        # Bus20 旧 con（馈线未清零）
└── mag_sweep_bus20_var.pkl    # Bus20 旧 var（馈线未清零）
```

---

## 三、运行环境

- **Python**：`C:/Users/Lenovo/.conda/envs/psse_312/python.exe`（Python 3.12，已装 psspy + numpy + scipy + numba）
  - ⚠️ 不要用 WorkBuddy 默认 managed env（缺 numba），必须用 psse_312。
- **工作目录**（PSSE 必须在目录内运行）：
  ```
  cd "D:/张欣然-F7D9CE/华为家庭存储/BUAA_research/BUAA2021/PSSE_Simulation"
  ```

---

## 四、关键代码机制（本次新增/确认）

1. **`dyradd` 参数（本次改的）**
   - `psse_dynamic.py` 的 `init_dynamic_case(...)` 和 `run_psse_simulation(...)` 新增可选 `dyradd=None`。
   - 默认（`None`）= 用原 stall dyr（`ieee39_RE_ieee39_RE_motorD_Stall_enabled.dyr`）。
   - 传 `dyradd='./Cases/IEEE39/..._noFdr20.dyr'` 即用馈线清零副本。

2. **辨识口径**
   - `scipy.optimize.least_squares(method='trf')` + 解析 Jacobian，5 次随机初值取 cost 最小。
   - `trim_k = n//20`（头尾各裁 5%）。
   - 核内 `a = para[0]/0.75, H2 = para[2]/0.75`，但返回的 `x[0]` 本身就是物理 `a`，故 `relerr_signed(x, truth)[0]` 直接比正确，**不需要再除以 0.75**。

3. **采样/真值换算**
   - con 用 A 组真值（`P_ma∈[0.3,0.7], X_a, X1_a, T0_a, H2_a`）。
   - var 用 B 组真值（`P_mb∈[0.3,0.7], X_b, X1_b, T0_b, H2_b`）。
   - `a=(X/X1-1)/T0/X1, b=X/X1/T0, H2, P_im=P_ma(或 P_mb)`。

4. **动态通道语义（查数据用）**
   - `BUS20 UD` / `BUS20 UQ`（负荷端口/辨识电压 d-q 分量），`|UD+jUQ|` 是辨识用端口电压。
   - `BUS20 LOAD BUS V`（高压侧负荷母线）与 `|UD+jUQ|` 数值在强补偿前相等。

---

## 五、已确认的结论（截至 2026-10-30）

**馈线阻抗是 a 偏小的主因（决定性）**：

| 场景 | a 偏差 | b 偏差 | core |
|---|---|---|---|
| Bus20 馈线未清零（base） | −51.7% | −27.5% | 5.74% |
| Bus20 馈线清零（noFdr） | **+4.5% ~ +10%** | +1.5% ~ +4% | ~1% |
| Bus3/4 历史正常量级 | ~+11.6% | 正常 | ~1% |

- 电压抬压实验为**否定结果**（抬压后 a 只从 −51.7% 微减到 −40%），电压不是主因。
- 馈线清零后 a 从 −51.7% 跳回 +4.5%，con/var 两条链路均验证通过。

---

## 六、注意事项 / 踩坑

1. **网络盘（华为家庭存储）文件锁**：偶发 `EBUSY` / `PSSE API save 失败 ierr=3`（写 cnv 锁冲突），`sleep 2~3s` 重试即成功；`one_time_simulation` 自带 `max_retries=8` 已覆盖。
2. **后台任务会被系统回收**：WorkBuddy 后台进程在会话/机器重启后被回收，无法跨天持续跑。已改用 **Windows 计划任务 `Bus20Sweep`**（开机自启 + 崩溃重启 3 次）独立跑扫描。
3. **QL ≤ −525 塌陷**：`load_chng_6` 过度无功补偿时，CMLDBLU2 内部 `BUS20 UD/UQ` 骤变精确 0.0000（内部参考系退化），系统级 `VOLT 20` 仍单调正常。抬压档位只用 −300/−400/−500，不要越过 −525。
4. **脚本已搬离 Temp**：辨识/扫描脚本已搬到项目 `scripts/` 目录，路径改为自动推断（`_HERE` → `ROOT`），输出统一到 `sweep_output/`，跨机器可移植（详见 `../跟进文档_另一台电脑接手.md`）。
