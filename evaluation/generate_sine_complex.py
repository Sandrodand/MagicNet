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

c2_complex_range = list(np.linspace(-0.25, -0.15, int(EL_PER_RANGE / 2))) + list(
    np.linspace(0.15, 0.25, int(EL_PER_RANGE / 2))
)
c3_complex_range = list(np.linspace(-2.2, -1.8, int(EL_PER_RANGE / 2))) + list(
    np.linspace(1.8, 2.2, int(EL_PER_RANGE / 2))
)

make_dir(f"outputs")
orig_stdout = sys.stdout
f = open(f"outputs/gen_sine_complex.txt", "w", buffering=1)
sys.stdout = f

try:
    cont = 1
    c1 = 0.5
    for c2 in c2_complex_range:
        for c3 in c3_complex_range:
            print(c2, c3)
            rw = RandomWalkGenerator(
                generator="sine",
                sine_classification_function=2,
                sine_c1=c1,
                sine_c2=c2,
                sine_c3=c3,
                max_consecutive_labels=MCL,
            )
            df = rw.generate(N, target_dependencies=TARGET_DEP, line_end="\n")
            with open(
                os.path.join(
                    ROOT_PATH, f"sine_rw{MCL}_mode{TARGET_DEP}_complex_n{cont}_+.pkl"
                ),
                "wb",
            ) as f:
                pickle.dump(rw, f)
            df.to_csv(
                os.path.join(
                    ROOT_PATH, f"sine_rw{MCL}_mode{TARGET_DEP}_complex_n{cont}_+.csv"
                ),
                index=False,
            )

            if MINUS:
                rw = RandomWalkGenerator(
                    generator="sine",
                    sine_classification_function=3,
                    sine_c1=c1,
                    sine_c2=c2,
                    sine_c3=c3,
                    max_consecutive_labels=MCL,
                )
                df = rw.generate(N, target_dependencies=TARGET_DEP)
                with open(
                    os.path.join(
                        ROOT_PATH,
                        f"sine_rw{MCL}_mode{TARGET_DEP}_complex_n{cont}_-.pkl",
                    ),
                    "wb",
                ) as f:
                    pickle.dump(rw, f)
                df.to_csv(
                    os.path.join(
                        ROOT_PATH,
                        f"sine_rw{MCL}_mode{TARGET_DEP}_complex_n{cont}_-.csv",
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
                        ROOT_PATH,
                        f"sine_rw{MCL}_mode{TARGET_DEP}_complex_n{cont}_-.csv",
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
