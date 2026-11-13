# PSS/E 电机混合负荷 8 参数辨识（团队共享项目）

本项目对 IEEE39 系统的电机混合负荷（CMLD 复合负荷，含 ZIP + 电机 A/B）做 **8 参数辨识**：`[a_a, b_a, H2_a, P_ima, a_b, b_b, H2_b, P_imb]`。

> 这是张老师课题组的共享仓库。**第一次上手请看本文档，动手前务必先读项目根目录的 `研究结论.md`（A 科学发现 / B 方法论 / C 环境踩坑）和 `工作日志.md`（状态表 + 倒序日志）。**

---

## 一、学生第一次上手（3 步）

### 第 1 步：装 WorkBuddy

到官网 **https://www.workbuddy.cn** 下载并安装 WorkBuddy 桌面端，用你的账号登录。

### 第 2 步：克隆代码

打开终端（或 WorkBuddy 自带终端），执行：

```bash
git clone https://github.com/zhangxinran07/PSSE_simulation
```

> 本仓库是公开的，克隆和拉取更新**不需要登录、不需要授权**。

### 第 3 步：用 WorkBuddy 打开项目目录

在 WorkBuddy 里**打开刚克隆下来的 `PSSE_simulation` 文件夹**（不是打开单个文件）。WorkBuddy 会自动识别并加载项目自带的技能（skill），之后你就能像张老师一样直接说「帮我生成一组仿真数据」「辨识这 8 个参数」，它会自动按项目约定去跑。

---

## 二、技能（skill）是怎么共享的

你**不需要手动安装任何东西**。项目里已经带了一份技能文件：

```
.workbuddy/skills/psse-simulation/SKILL.md
```

它随 git 仓库一起分发。你只要用 WorkBuddy 打开这个项目目录，WorkBuddy 就会**自动加载**这份技能——里面写清了：

- PSS/E 装在哪、用哪个 Python 解释器（**每台电脑去自己的地方找 PSS/E，别只查 C 盘就断言没装**）；
- 怎么生成仿真数据、怎么辨识、怎么换观测母线；
- 辨识口径（`a = para/0.75`、`H2 = para/0.75`，别混，否则差 0.75 倍）；
- 前人踩过的坑（`.out` 文件用「截断」别用「删除」、`.dyr` 改动前必备份、别双机并行跑同一 case 目录等）。

**技能以后更新了，你只要 `git pull` 就能同步到最新版。**

---

## 三、GitHub 同步代码

### 拉取最新代码（日常必做）

```bash
cd PSSE_simulation
git pull
```

### 提交你的修改

```bash
git add .
git commit -m "写清楚你改了什么"
git push
```

> **push 需要仓库写权限**：把你的 GitHub 账号告诉张老师，让他在仓库 `Settings → Collaborators` 里把你加进去（或者你们组建一个 GitHub 组织统一管理）。只有 clone / pull 不需要权限。

### 提交前请注意

- **大文件别入库**：`.pkl`、`.out`、`.log`、`_cnv.sav`、`conec.for`、`conet.for` 等已在 `.gitignore` 里排除，正常 `git add .` 不会误提交。若确有重要结果要分享，单独跟张老师确认。
- **改动 `.dyr` 前必备份并知会张老师**（历史上有 Lpp>Lp 非法参数导致数值发散的事故）。
- 提交信息写中文、写清楚「改了什么」，方便张老师 review。

---

## 四、环境要求（跑仿真前确认）

1. **Python 3.12**（对应 PSS/E 的 `PSSPY312`）。系统默认 `python` 往往是 3.13，会 `import psse36/psspy` 失败。先 `where python` 找到你机器上的 3.12 解释器。
2. **PSS/E 已安装**，且用 `package/psse_dynamic.py` 头部的 `_find_psspy()` 自动探测安装路径（**每台电脑去自己的地方找，别硬编码**）。
3. 案例文件在 `Cases/IEEE39/`，载入用**相对路径 + 正斜杠**。

详细环境约定见 `研究结论.md` 的 **C 章**。

---

## 五、核心入口速查

| 想做什么 | 去哪 / 用什么 |
|---|---|
| 生成仿真数据 | `Sample_Generation/main_simu.py` → `package/psse_dynamic.one_time_simulation(...)` |
| 辨识 8 参数 | `package/identification.py` 的 `identify_mix(...)` / `identify_multistart(...)` |
| 换观测母线 / 扰动对齐 | `scripts/bus_switch_probe.py`、`probe_ambient.py` |
| 可辨识性诊断 | `scripts/diagnose_identifiability.py`、`scripts/control_zero_mismatch.py` |
| 看结论和踩坑 | `研究结论.md`（A/B/C 三章）、`工作日志.md` |

---

## 六、遇到问题

- 先翻 `研究结论.md` 的 **C 章（环境/运维踩坑）** 和 `工作日志.md`，大概率前人已经踩过并记了解决方案。
- 仍解决不了，在项目里直接问 WorkBuddy（它会读到技能里的经验），或找张老师。
