from dsm import datasets
import numpy as np
from dsm import DeepSurvivalMachines
import torch 
SEED = 1
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
                            distribution = 'LogNormal',
                            layers = [128],
                            device = device)
# The fit method is called to train the model
model.fit(x_train, t_train, e_train, iters = 50, batch_size = 4096, learning_rate = 1e-3)
out_survival = model.predict_survival(x_test, times)
print(out_survival)