# -*- coding: utf-8 -*-
"""统计海量数据的实际电压扰动范围（max|u| - min|u|）。

背景：5000 组用 disturbance_type='ambient_a' + disturbance_range=500（MW 直给，不走
      ambient_u/ambient_p 的达标校验）。本脚本事后统计每组实际电压扰动范围，确认
      虽用 MW 直给、但各组电压扰动幅度是否稳定一致（真值参数不同会引入波动）。

口径（与 psse_dynamic._measure_achieved_range / B.6 标定表一致）：
    电压范围 = max(|u|) - min(|u|)      (u = ud + 1j*uq, 母线电压)
    相对扰动 = (max|u|-min|u|) / mean(|u|)
    B.6 标定: bus3 500MW -> 电压 max-min ≈ 0.023 p.u.

用法：
    python scripts/b5k_voltage_range.py              # 统计 b5k_ori_shard*.pkl（5000 组，增量）
    python scripts/b5k_voltage_range.py --old        # 统计 b_line_ori_shard*.pkl（1000 组，基线对照）

输出：Sample_Generation/results/<prefix>_voltage_range.txt
"""
import os
import sys
import glob
import pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
RES = os.path.join(ROOT, 'Sample_Generation', 'results')


def collect(prefix):
    files = sorted(glob.glob(os.path.join(RES, prefix + '*.pkl')))
    if not files:
        print('未找到 %s*.pkl' % prefix)
        return None, None, []
    ranges, rels = [], []
    per_shard = []
    for f in files:
        rec = pickle.load(open(f, 'rb'))
        sid = rec['shard']
        rs, rl = [], []
        for g in rec['groups']:
            mag = np.abs(np.asarray(g['u'], dtype=complex))
            r = float(mag.max() - mag.min())
            rel = float(r / mag.mean()) if mag.mean() > 0 else 0.0
            rs.append(r)
            rl.append(rel)
        ranges.extend(rs)
        rels.extend(rl)
        per_shard.append((sid, np.array(rs)))
    return np.array(ranges), np.array(rels), per_shard


def report(name, ranges, rels, per_shard):
    print('=' * 70)
    print('%-10s  n = %d' % (name, len(ranges)))
    print('=' * 70)
    if len(ranges) == 0:
        return
    for label, arr in [('电压范围 |u| max-min', ranges),
                       ('相对扰动 (max-min)/mean', rels)]:
        print('  %-24s 中位 %.4f  均值 %.4f  std %.4f' % (
            label, np.median(arr), np.mean(arr), np.std(arr)))
        print('  %-24s  min %.4f  p5 %.4f  p95 %.4f  max %.4f' % (
            '', arr.min(), np.percentile(arr, 5), np.percentile(arr, 95), arr.max()))
    print()
    if per_shard:
        print('  各分片电压范围中位: ' + '  '.join(
            'sh%d=%.4f' % (sid, np.median(rs)) for sid, rs in sorted(per_shard)))
    print('  对照 B.6 标定: bus3 500MW -> 电压 max-min ≈ 0.023 p.u.')
    print()


def main():
    old = '--old' in sys.argv
    if old:
        prefix, out = 'b_line_ori_shard', 'b_line_voltage_range.txt'
    else:
        prefix, out = 'b5k_ori_shard', 'b5k_voltage_range.txt'
    ranges, rels, per_shard = collect(prefix)
    if ranges is None:
        return
    report(prefix, ranges, rels, per_shard)
    # 落盘
    with open(os.path.join(RES, out), 'w', encoding='utf-8') as f:
        f.write('%s  n=%d\n' % (prefix, len(ranges)))
        f.write('电压范围 |u| max-min: 中位 %.4f 均值 %.4f std %.4f  min %.4f p5 %.4f p95 %.4f max %.4f\n'
                % (np.median(ranges), np.mean(ranges), np.std(ranges),
                   ranges.min(), np.percentile(ranges, 5), np.percentile(ranges, 95), ranges.max()))
        f.write('相对扰动 (max-min)/mean: 中位 %.5f 均值 %.5f std %.5f\n'
                % (np.median(rels), np.mean(rels), np.std(rels)))
        if per_shard:
            f.write('各分片电压范围中位: ' + '  '.join(
                'sh%d=%.4f' % (sid, np.median(rs)) for sid, rs in sorted(per_shard)) + '\n')
        f.write('对照 B.6 标定: bus3 500MW -> 电压 max-min ≈ 0.023 p.u.\n')
    print('已写入 %s' % os.path.join(RES, out))


if __name__ == '__main__':
    main()
