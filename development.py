from dsm import datasets
import numpy as np
from dsm import DeepSurvivalMachines, DeepRecurrentSurvivalMachines
import torch

SEED = 2
torch.manual_seed(SEED)
import random

random.seed(SEED)
np.random.seed(SEED)

# x, t, e = datasets.load_dataset("SUPPORT")
x, t, e = datasets.load_dataset("PBC", sequential=True)
horizons = [0.25, 0.5, 0.75]
n = len(x)

times = np.quantile([t_[-1] for t_, e_ in zip(t, e) if e_[-1] == 1], horizons).tolist()

tr_size = int(n * 0.70)
vl_size = int(n * 0.10)
te_size = int(n * 0.20)

x_train, x_test, x_val = (
    np.array(x[:tr_size], dtype=object),
    np.array(x[-te_size:], dtype=object),
    np.array(x[tr_size : tr_size + vl_size], dtype=object),
)
t_train, t_test, t_val = (
    np.array(t[:tr_size], dtype=object),
    np.array(t[-te_size:], dtype=object),
    np.array(t[tr_size : tr_size + vl_size], dtype=object),
)
e_train, e_test, e_val = (
    np.array(e[:tr_size], dtype=object),
    np.array(e[-te_size:], dtype=object),
    np.array(e[tr_size : tr_size + vl_size], dtype=object),
)

device = torch.device("cuda:1" if torch.cuda.is_available() == True else "cpu")
# model = DeepSurvivalMachines(
#     discount=0.5,
#     temp=1,
#     k=200,
#     #distribution = 'Weibull',
#     distribution="LogNormal",
#     layers=[128, 128],
#     device=device,
# )
model = DeepRecurrentSurvivalMachines(
    discount=0.5,
    temp=1,
    k=3,
    distribution="Weibull",
    typ="LSTM",
    layers=1,
    hidden=64,
    device=device,
)

# The fit method is called to train the model
model.fit(
    x_train,
    t_train,
    e_train,
    iters=50,
    batch_size=2048,
    learning_rate=1e-2,
    weight_decay=1e-3,
    random_state=SEED,
)
out_survival, std = model.predict_survival(x_test, times)
out_risk = model.predict_risk(x_test, times)
print(out_survival, std)
from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc

cis = []
brs = []

# et_train = np.array(
#     [(e_train[i], t_train[i]) for i in range(len(e_train))],
#     dtype=[("e", bool), ("t", float)],
# )
# et_test = np.array(
#     [(e_test[i], t_test[i]) for i in range(len(e_test))],
#     dtype=[("e", bool), ("t", float)],
# )
# et_val = np.array(
#     [(e_val[i], t_val[i]) for i in range(len(e_val))], dtype=[("e", bool), ("t", float)]
# )

et_train = np.array(
    [
        (e_train[i][j], t_train[i][j])
        for i in range(len(e_train))
        for j in range(len(e_train[i]))
    ],
    dtype=[("e", bool), ("t", float)],
)
et_test = np.array(
    [
        (e_test[i][j], t_test[i][j])
        for i in range(len(e_test))
        for j in range(len(e_test[i]))
    ],
    dtype=[("e", bool), ("t", float)],
)
et_val = np.array(
    [
        (e_val[i][j], t_val[i][j])
        for i in range(len(e_val))
        for j in range(len(e_val[i]))
    ],
    dtype=[("e", bool), ("t", float)],
)

for i, _ in enumerate(times):
    cis.append(concordance_index_ipcw(et_train, et_test, out_risk[:, i], times[i])[0])
brs.append(brier_score(et_train, et_test, out_survival, times)[1])
roc_auc = []
for i, _ in enumerate(times):
    roc_auc.append(
        cumulative_dynamic_auc(et_train, et_test, out_risk[:, i], times[i])[0]
    )
for horizon in enumerate(horizons):
    print(f"For {horizon[1]} quantile,")
    print("TD Concordance Index:", cis[horizon[0]])
    print("Brier Score:", brs[0][horizon[0]])
    print("ROC AUC ", roc_auc[horizon[0]][0], "\n")
