import traceback

from evaluation.prequential_evaluation import make_dir

ROOT_PATH = "datasets/federated/sine"
N = 75000
MCL = 10
TARGET_DEP = 5
EL_PER_RANGE = 4
MINUS = False

import pickle
from data_gen.rw_generator import *
import os
import sys

c1_simple_range = [1, 1, 0, 0]
c2_simple_range = [-1, -1, 1, 1]
c3_simple_range = list(np.linspace(0.8, 1.2, EL_PER_RANGE))

make_dir(f"outputs")
orig_stdout = sys.stdout
f = open(f"outputs/gen_sine_simple.txt", "w", buffering=1)
sys.stdout = f

try:
    cont = 1
    for c1, c2 in zip(c1_simple_range, c2_simple_range):
        for c3 in c3_simple_range:
            print(c1, c2, c3)
            rw = RandomWalkGenerator(
                generator="sine",
                sine_classification_function=0,
                sine_c1=c1,
                sine_c2=c2,
                sine_c3=c3,
                max_consecutive_labels=MCL,
            )
            df = rw.generate(N, target_dependencies=TARGET_DEP, line_end=None)
            with open(
                os.path.join(
                    ROOT_PATH, f"sine_rw{MCL}_mode{TARGET_DEP}_simple_n{cont}_+.pkl"
                ),
                "wb",
            ) as f:
                pickle.dump(rw, f)
            df.to_csv(
                os.path.join(
                    ROOT_PATH, f"sine_rw{MCL}_mode{TARGET_DEP}_simple_n{cont}_+.csv"
                ),
                index=False,
            )

            if MINUS:
                rw = RandomWalkGenerator(
                    generator="sine",
                    sine_classification_function=1,
                    sine_c1=c1,
                    sine_c2=c2,
                    sine_c3=c3,
                    max_consecutive_labels=MCL,
                )
                df = rw.generate(N, target_dependencies=TARGET_DEP, line_end=None)
                with open(
                    os.path.join(
                        ROOT_PATH, f"sine_rw{MCL}_mode{TARGET_DEP}_simple_n{cont}_-.pkl"
                    ),
                    "wb",
                ) as f:
                    pickle.dump(rw, f)
                df.to_csv(
                    os.path.join(
                        ROOT_PATH, f"sine_rw{MCL}_mode{TARGET_DEP}_simple_n{cont}_-.csv"
                    ),
                    index=False,
                )
            else:
                df_minus = df.copy()
                df_minus["target"] = df_minus["target"].apply(
                    lambda x: 1 if x == 0 else 0
                )
                df_minus["classification"] = df_minus["classification"].apply(
                    lambda x: 1 if x == 0 else 0
                )
                df_minus.to_csv(
                    os.path.join(
                        ROOT_PATH, f"sine_rw{MCL}_mode{TARGET_DEP}_simple_n{cont}_-.csv"
                    ),
                    index=False,
                )
            cont += 1
            print()
except Exception:
    print(traceback.format_exc())
    sys.stdout = orig_stdout
    f.close()
    print(traceback.format_exc())
print("\n\nEND.")
sys.stdout = orig_stdout
f.close()
