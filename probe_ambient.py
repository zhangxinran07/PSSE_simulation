import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "Sample_Generation"))

import main_simu as ms


def probe(busid, mags, total_time=30, repeats=3):
    """对指定辨识母线, 逐档跑 ambient 仿真, 实测电压与功率波动。

    每档重复 repeats 次取中位。返回 DataFrame:
      mag / u_std / u_range / u_min / u_max / p_std / p_range
    其中 p_range = (P.max - P.min) / P.mean, 即静态工作点附近的标幺功率波动范围。
    探针不修改电机参数, 因为电压/功率波动幅度主要由网络决定。
    """
    ms.kwargs["load_bus_monitor"] = [busid]
    ms.kwargs["ld_con_chng_buses"] = []
    ms.kwargs["ld_con_chng_values"] = []
    ms.kwargs["total_time"] = total_time

    rows = []
    for mag in mags:
        ms.kwargs["disturbance_type"] = "ambient_a"
        ms.kwargs["disturbance_range"] = mag
        agg = {"u_std": [], "u_range": [], "u_min": [], "u_max": [], "p_std": [], "p_range": []}
        for _ in range(repeats):
            df, d_para, t = ms.ps_d.one_time_simulation(ms.kwargs)
            cols = list(df.columns)
            ud = df.values[:, [i for i in range(len(cols)) if ("BUS%d UD" % busid) in cols[i]]].reshape(len(df))
            uq = df.values[:, [i for i in range(len(cols)) if ("BUS%d UQ" % busid) in cols[i]]].reshape(len(df))
            P = df.values[:, [i for i in range(len(cols)) if ("PLOD %d[" % busid) in cols[i]]].reshape(len(df))
            u = np.abs(ud + 1j * uq)
            pmean = P.mean()
            agg["u_std"].append(u.std())
            agg["u_range"].append(u.max() - u.min())
            agg["u_min"].append(u.min())
            agg["u_max"].append(u.max())
            agg["p_std"].append(P.std() / pmean)
            agg["p_range"].append((P.max() - P.min()) / pmean)
        rows.append({
            "mag": mag,
            "u_std": float(np.median(agg["u_std"])),
            "u_range": float(np.median(agg["u_range"])),
            "u_min": float(np.median(agg["u_min"])),
            "u_max": float(np.median(agg["u_max"])),
            "p_std": float(np.median(agg["p_std"])),
            "p_range": float(np.median(agg["p_range"])),
        })
        print("mag=%5d  u_range=%.5f  p_range=%.5f  u_min=%.4f  u_max=%.4f  (%.1fs)" %
              (mag, rows[-1]["u_range"], rows[-1]["p_range"], rows[-1]["u_min"], rows[-1]["u_max"], t),
              flush=True)
    return pd.DataFrame(rows)


def _append_calib(csv_name, out, busid, key):
    path = os.path.join(ROOT, csv_name)
    if os.path.exists(path):
        cal = pd.read_csv(path)
    else:
        cal = pd.DataFrame(columns=["bus", "range", "mag"])
    for _, row in out.iterrows():
        mask = (cal["bus"] == busid) & (cal["mag"] == int(row["mag"]))
        if mask.any():
            cal.loc[mask, "range"] = float(row[key])
        else:
            cal = pd.concat([cal, pd.DataFrame([{"bus": busid, "range": float(row[key]), "mag": int(row["mag"])}])],
                            ignore_index=True)
    cal = cal.sort_values(["bus", "range"]).reset_index(drop=True)
    cal.to_csv(path, index=False)
    print("updated:", path)


def main():
    busid = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    mags = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [500, 1000, 2000, 3000]
    repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    out = probe(busid, mags, repeats=repeats)
    out_path = os.path.join(ROOT, "probe_ambient_bus%d.csv" % busid)
    out.to_csv(out_path, index=False)
    print("saved:", out_path)

    _append_calib("ambient_u_calib.csv", out, busid, "u_range")
    _append_calib("ambient_p_calib.csv", out, busid, "p_range")


if __name__ == "__main__":
    main()
