# 另一台电脑并行跑 var 扫描 — 操作说明

> 给另一台装了 PSSE 的电脑用。目标：和这台机器并行，把这台跑 con、那台跑 var，
> 加速 Bus20 馈线清零（noFdr）全幅值扫描。
>
> ⚠️ **PSSE 路径每台电脑各找各的**：本机（D盘主电脑）在 `D:\Program Files\PTI\`；本文档里若出现 `C:\Program Files\PTI\...` 是别的机器的写法，接手电脑请改用**自己机器上**的 PTI 目录与 Python 版本匹配的 PSSPY 子目录。详见《研究结论.md》C.8。

---

## 一、分工

| 机器 | 任务 | 量 | 预计耗时 |
|---|---|---|---|
| 这台（当前） | con（恒转矩）剩余 | ~195 条 | ~7 小时 |
| 另一台 | var（变转矩）全部 | 300 条 | ~10 小时 |

两者跑完后，结果都落在 NAS 的 `sweep_output/` 目录，直接汇总对比，无需手动拼。

---

## 二、另一台电脑操作步骤

### 第 1 步：确认 Python 环境

在另一台电脑的命令行（cmd 或 PowerShell）里执行：

```
python -c "import psspy, numpy, scipy, numba; print('OK', numpy.__version__, scipy.__version__, numba.__version__)"
```

- 必须能 `import` 这 4 个包：`psspy`（PSSE 接口）、`numpy` / `scipy` / `numba`（辨识依赖）。
- 缺 numba/scipy 就 `pip install numpy scipy numba`。
- 记下这个 python 的完整路径（`where python` 查看，后面要用）。

### 第 2 步：把项目复制到本地盘（关键，别在 NAS 目录直接跑）

⚠️ PSSE 会在 `Cases/IEEE39/` 写固定名的临时文件（`IEEE39_RE.sav`、`IEEE39_RE_cnv.sav`、`IEEE39_RE.snp`、`IEEE39_RE.out`）。
两台电脑如果都对着 NAS 同一目录跑，会争抢这批同名文件、互相覆盖。所以另一台必须在**自己的本地副本**里跑。

复制命令（把 `<NAS盘符>` 换成那台电脑上 NAS 映射的盘符，`D:\PSSE_local` 可自定）：

```
robocopy "<NAS盘符>\张欣然-F7D9CE\华为家庭存储\BUAA_research\BUAA2021\PSSE_Simulation" "D:\PSSE_local" /E /XD __pycache__ psse_stubs
```

或者用文件管理器，只需复制这三个文件夹到本地：

- `Cases`（39M）
- `package`（1.9M）
- `scripts`（88K）

### 第 3 步：跑 var 扫描

```
cd /d D:\PSSE_local
python scripts\bus20_nofdr_sweep.py var
```

- 会逐条打印进度：`[xx/300] mag=xxx g=xx |u|=... a_sgn=... b_sgn=...`。
- **断点续跑**：每条跑完立即落盘，中途断了重跑同一条命令即可，已完成的不重复。

### 第 4 步：回传结果

跑完后，把本地结果复制回 NAS：

```
copy /Y "D:\PSSE_local\sweep_output\bus20_nofdr_sweep_var.pkl" "<NAS盘符>\张欣然-F7D9CE\华为家庭存储\BUAA_research\BUAA2021\PSSE_Simulation\sweep_output\"
```

---

## 三、注意事项

1. **license**：如果跑起来报 license 相关错误，说明 PSSE 授权是单机绑定、这台不能同时用，需要单独确认授权方式（单机锁 vs 网络浮动授权）。
2. **断点续跑**：脚本每条跑完立即写 pkl，断了重跑即可，进度不丢。
3. **脚本路径已改**：`scripts/` 下的三个脚本已改成「自动推断路径」，不管 NAS 映射成什么盘符、本机用户名是什么，都能直接跑，无需再改任何路径。
