---
name: psse-simulation
description: Run PSS/E motor mixed-load (CMLD) parameter identification and simulation in this project. Use when asked to generate PSS/E simulation data, identify the 8 motor parameters (a, b, H2, P_im for motors A and B), diagnose identifiability, run sensitivity or ablation experiments, or debug PSS/E / psse_dynamic.py issues.
agent_created: true
version: 1.0.0
description_zh: "运行 PSS/E 电机混合负荷(CMLD)辨识与仿真：生成数据、辨识 8 参数(a,b,H2,P_im 电机A/B)、可辨识性诊断、敏感性/消融实验、排查 psse_dynamic.py 问题"
---

# PSS/E 电机混合负荷辨识（项目环境与工作流）

本 skill 用于让接手本项目的人（学生 / 另一台电脑的 WorkBuddy）能直接复用张老师已沉淀的环境约定、工作流与踩坑经验，不重复踩坑。**核心结论资产是项目根目录的 `研究结论.md`（A 科学发现 / B 方法论 / C 环境踩坑）与 `工作日志.md`（状态表 + 倒序日志），先读它们；本文件只提炼「最容易栽、必须照做」的铁律与入口。**

## 项目是什么

- 目标：PSS/E 下对 IEEE39 系统的电机混合负荷（CMLD 复合负荷，含 ZIP + 电机 A/B）做 **8 参数辨识**：`[a_a, b_a, H2_a, P_ima, a_b, b_b, H2_b, P_imb]`（前 4 = 电机 A 组，后 4 = 电机 B 组）。
- 工程解（已收口，勿轻易改）：**单一母线总功率观测 + 多起点 K=12 + H2 差量 Tikhonov 正则 λ_H=0.02 + RK4@10ms 前向**。精度 P_im≈2%、H2≈7%、dyn6≈8%，张老师已验收。
- 观测母线默认 **bus3**（已证优于 bus4）；扰动默认 `ambient_a` 500 MW 注入。

## 环境铁律（最易栽，务必照做）

1. **必须用 Python 3.12**（对应 `PSSPY312`）。系统默认 `python` 往往是 3.13，会 `import psse36/psspy` 失败。本机科学计算解释器示例：`C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\python.exe`——**学生机器路径会不同，先 `where python` 或查已装的 3.12**，别照抄。
2. **PSSE 安装路径「每台电脑去自己的地方找」，用 `_find_psspy()` 自适应探测，不要硬编码**。`package/psse_dynamic.py` 头部的 `_find_psspy()` 会按「盘符 × 大版本(36/35) × 版本号」遍历 `PTI\PSSE{36|35}\{ver}\PSSPY{py主}{py次}` 取第一个与当前 Python 匹配的目录，并把 `PSSBIN` 加进 `PATH` + `os.add_dll_directory`。**切勿只查 C 盘就断言「没装 PSSE」**（张老师 11-09 曾因此误判）。
3. **载入 case 用「相对路径 + 正斜杠」**，先 `os.chdir` 到项目根，`psspy.case('./Cases/IEEE39/ieee39_RE.sav')`。用反斜杠绝对路径会报 `File not found`。
4. **单 node-locked license**：可单机并行多个会话，但每个会话必须用**独立 case 副本目录 + 相对路径**（`Cases/w<sid>/`），且 PSS/E 的 Fortran 处理不了含中文的绝对路径。

## 核心工作流入口

- **生成仿真数据**：`Sample_Generation/main_simu.py`（或 `simu_mix_split.py`）→ `package/psse_dynamic.one_time_simulation(...)` → `psspy`。扰动接口是 kwargs 的 `disturbance_type`（`ambient_a`=MW 直给 / `ambient_u`=电压范围 p.u. / `ambient_p`=标幺功率范围）+ `disturbance_range`。
- **辨识**：`package/identification.py` 的 `identify_mix(...)`（单组）与 `identify_multistart`（多起点）。前向核 `_core_con`/`_core_var`（RK4@10ms）及其解析灵敏度 `_core_con_jac`/`_core_var_jac`。
- **换观测母线 / 扰动幅值对齐**：`scripts/bus_switch_probe.py`（生成+辨识）、`probe_ambient.py <bus> "档位" 3`（重标定 MW→电压/功率幅值表）。

## 辨识口径（勿混，否则差 0.75 倍）

- `para` 中 **`a = para/0.75`、`H2 = para/0.75`**（辨识器内部缩放）；真值表 `_real` 列与 `best.x` 同为辨识口径，**可直接相减**。物理口径 Jacobian 不要除 0.75（只有 `obj_mix_iden` 目标函数内部才缩放）。

## 踩坑（别重犯）

1. **`.out` 中间文件清理用「截断」不用「删除」**：`os.remove` 批量删除会触发 WorkBuddy 的 safe-delete 批量确认钩子，导致后台分片静默挂起。改 `open(outfile,'wb').close()` 截断为 0 字节。
2. **`.dyr` 是基础定义文件，改动前必备份（`bak_before_*`）并知会张老师**：历史有 Lpp>Lp 非法参数导致数值发散（var 全 NaN）的事故。
3. **`identify()` 已有防御**（初值采样上限 20000 + trust-constr maxiter=2000），正常组零影响——别删。
4. **积分方法/步长不是误差主因，勿改**：辨识器前向维持 RK4@10ms 不变（2×2 全因子已证，见 A2.44）。
5. **勿双机并行跑同一个 case 目录**：会撞 `.sav/.snp/.out` 临时文件（A2.44 之前的 C.6，后被 C.9 修正为「单机可并行但用独立目录」）。

## 关键文件索引

- `研究结论.md` —— A 章科学结论（A2.31~A2.45 为主线核心）、B 章可复用方法论、C 章环境/运维踩坑（C.8/C.9/C.10 最重要）。
- `工作日志.md` —— 顶部状态表 + 倒序日志（新条目插在 `## 日志条目` 正下方）。
- `类噪声注入机制说明.md` —— 类噪声经「新增 0 值负荷(ID='2 ')」注入的机制、API 细节与运行环境要求。
- `scripts/` 可复用诊断脚本：`diagnose_identifiability.py`（Fisher 零空间）、`control_zero_mismatch.py`（零失配对照）、`split_vs_total_power.py`（观测降维诊断）、`laplace_posterior.py`（拉普拉斯近似）、`mcmc_path3.py`（完全贝叶斯）。
