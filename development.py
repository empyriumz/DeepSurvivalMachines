from dsm import datasets
import numpy as np
from dsm import DeepSurvivalMachines
import torch 
SEED = 2
torch.manual_seed(SEED)
import random
random.seed(SEED)
np.random.seed(SEED)

x, t, e = datasets.load_dataset('SUPPORT')
horizons = [0.25, 0.5, 0.75]
times = np.quantile(t[e==1], horizons).tolist()

n = len(x)

tr_size = int(n*0.70)
vl_size = int(n*0.10)
te_size = int(n*0.20)

x_train, x_test, x_val = x[:tr_size], x[-te_size:], x[tr_size:tr_size+vl_size]
t_train, t_test, t_val = t[:tr_size], t[-te_size:], t[tr_size:tr_size+vl_size]
e_train, e_test, e_val = e[:tr_size], e[-te_size:], e[tr_size:tr_size+vl_size]

device = torch.device("cuda:1" if torch.cuda.is_available() == True else "cpu")
model = DeepSurvivalMachines(k = 3,
                            distribution = 'Weibull',
                            #distribution = 'LogNormal',
                            layers = [128, 128],
                            device = device)
# The fit method is called to train the model
model.fit(x_train, t_train, e_train, iters = 50, batch_size = 2048, learning_rate = 1e-2, weight_decay=1e-3)
out_survival, std = model.predict_survival(x_test, times)
out_risk = model.predict_risk(x_test, times)
print(out_survival, std)
from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc

cis = []
brs = []

et_train = np.array([(e_train[i], t_train[i]) for i in range(len(e_train))],
                 dtype = [('e', bool), ('t', float)])
et_test = np.array([(e_test[i], t_test[i]) for i in range(len(e_test))],
                 dtype = [('e', bool), ('t', float)])
et_val = np.array([(e_val[i], t_val[i]) for i in range(len(e_val))],
                 dtype = [('e', bool), ('t', float)])

for i, _ in enumerate(times):
    cis.append(concordance_index_ipcw(et_train, et_test, out_risk[:, i], times[i])[0])
brs.append(brier_score(et_train, et_test, out_survival, times)[1])
roc_auc = []
for i, _ in enumerate(times):
    roc_auc.append(cumulative_dynamic_auc(et_train, et_test, out_risk[:, i], times[i])[0])
for horizon in enumerate(horizons):
    print(f"For {horizon[1]} quantile,")
    print("TD Concordance Index:", cis[horizon[0]])
    print("Brier Score:", brs[0][horizon[0]])
    print("ROC AUC ", roc_auc[horizon[0]][0], "\n")