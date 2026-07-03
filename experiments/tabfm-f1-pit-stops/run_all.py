"""Driver: compute every cached run for the notebook, cheap CPU runs first.

Launch detached (survives the shell, resists idle sleep):
    nohup caffeinate -i .venv/bin/python run_all.py > artifacts/run.log 2>&1 &
"""
import time

import experiment as ex

t0 = time.time()
ex.log('=== run_all start ===')
ex.data()

# CatBoost (CPU, minutes) - early signal
for n in ex.SWEEP_CTX:
    ex.run_catboost('raw', n)
ex.run_catboost('engineered', ex.N_CTX_FE)
ex.run_catboost('raw')

# TabFM (GPU) - small contexts first, then the expensive engineered run
for n in ex.SWEEP_CTX:
    ex.run_tabfm('raw', n)
ex.run_tabfm('engineered', ex.N_CTX_FE)

# slowest CPU run last (tuned params, full fold, 104 cols)
ex.run_catboost('engineered', tuned=True)

ex.log(f'=== run_all done in {(time.time()-t0)/60:.0f} min ===')
