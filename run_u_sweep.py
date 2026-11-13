import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "Sample_Generation"))

import main_simu as ms


def main():
    u_range = float(sys.argv[1])
    n_groups = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    ms.kwargs["disturbance_type"] = "ambient_u"
    ms.kwargs["disturbance_range"] = u_range
    ms.simu_iden_mix(n_groups=n_groups)
    tagged = ms.result_path_mix.replace(".pkl", "_u%.3f.pkl" % u_range)
    shutil.copy(ms.result_path_mix, tagged)
    print("DONE_U", u_range, "->", tagged)


if __name__ == "__main__":
    main()
