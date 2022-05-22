## Code for Information-Theoretic Analysis of Epistemic Uncertainty in Bayesian Meta-learning

This repository contains code for "[Information-Theoretic Analysis of Epistemic Uncertainty in Bayesian Meta-learning](https://arxiv.org/pdf/2106.00252.pdf)" - Sharu Theresa Jose, Sangwoo Park, and Osvaldo Simeone

### Dependencies

This program is written in python 3.8 and uses PyTorch 1.8.1.

### Essential codes
- C-MINE with SMILE can be found in `utils/cmi.py`
- ME(M)R computation (eq 11, 13) can be found in `memr.py`. Detailed usage can be found below.
- Information-theoretic bounds computation (eq 15) can be found in `mi_para.py` (parameter-level sensitivity) and `mi_hyper.py` (hyperparameter-level sensitivity). Detailed usage can be found below.

### Bayesian NN Regression
#### 1) Fig. 2 (when meta-learning is highly effetive: variance of W is small)
- Data generation

    Execute `runs/small_std_W/total_data_gen.sh`
    For the default settings and other argument options, please refer to `datagen.py`
- MER:

    Execute `runs/small_std_W/memr_conven_over_m.sh`
    For the default settings and other argument options, please refer to `memr.py`
- MEMR:

    Execute `runs/small_std_W/memr_over_m.sh` and `runs/small_std_W/memr_over_N.sh`
    For the default settings and other argument options, please refer to `memr.py`
- Information-theoretic bounds (hyperparameter-level sensitivity):

    Execute `runs/small_std_W/mi_hyper_over_m.sh` and `runs/small_std_W/mi_hyper_over_N.sh` 
    For the default settings and other argument options, please refer to `mi_hyper.py`
- Information-theoretic bounds (parameter-level sensitivity):

    Execute `runs/small_std_W/mi_para_over_m.sh`
    For the default settings and other argument options, please refer to `mi_para.py`
#### 2) Fig. 3 (when meta-learning is less effetive: variance of W is high)
- change `/small_std_W/` to `/large_std_W/` in the Fig. 2 case.
