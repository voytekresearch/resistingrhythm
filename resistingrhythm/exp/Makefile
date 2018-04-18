SHELL=/bin/bash -O expand_aliases
# DATA_PATH=/Users/type/Code/resistingrhythm/data/
DATA_PATH=/home/ejp/src/resistingrhythm/data

create_data_path:
	-mkdir $(DATA_PATH)

# --------------------------------------------------------------
# Replicate 3 sitm, same params, different draws
stim0:
	python homeostasis_experiments.py create_stim --t=20 --t_onset=0 --w=20 --rate=6 --n=1000 --name=$(DATA_PATH)/stim0 --seed=23434
stim1:
	python homeostasis_experiments.py create_stim --t=20 --t_onset=0 --w=20 --rate=6 --n=1000 --name=$(DATA_PATH)/stim1 --seed=87
stim2:
	python homeostasis_experiments.py create_stim --t=20 --t_onset=0 --w=20 --rate=6 --n=1000 --name=$(DATA_PATH)/stim2 --seed=5318

# --------------------------------------------------------------
# Generate osc pops, at several different firing rates
# Sustained oscillations, all have f = 8
# 0.5 Hz - 6 Hz
osc0:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=0 --n_cycles=180 --rate=0.5 --n=1000 --name=$(DATA_PATH)/osc0 --seed=42
# 1 Hz
osc1:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=0 --n_cycles=180 --rate=1.0 --n=1000 --name=$(DATA_PATH)/osc1 --seed=42
# 2 Hz
osc2:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=0 --n_cycles=180 --rate=2.0 --n=1000 --name=$(DATA_PATH)/osc2 --seed=42
# 3 Hz
osc3:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=0 --n_cycles=180 --rate=3.0 --n=1000 --name=$(DATA_PATH)/osc3 --seed=42
# 4 Hz
osc4:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=0 --n_cycles=180 --rate=4.0 --n=1000 --name=$(DATA_PATH)/osc4 --seed=42
# 5 hz
osc5:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=0 --n_cycles=180 --rate=5.0 --n=1000 --name=$(DATA_PATH)/osc5 --seed=42
# 6 hz  (equal to stim == 'patho')
osc6:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=0 --n_cycles=180 --rate=6.0 --n=1000 --name=$(DATA_PATH)/osc6 --seed=42

# -
# 3 cycles of burst, starting at the end ('H' on for stim)
burst0:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=19.5 --n_cycles=3 --rate=0.5 --n=1000 --name=$(DATA_PATH)/burst0 --seed=42
# 1 Hz
burst1:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=19.5 --n_cycles=3 --rate=1.0 --n=1000 --name=$(DATA_PATH)/burst1 --seed=42
# 2 Hz
burst2:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=19.5 --n_cycles=3 --rate=2.0 --n=1000 --name=$(DATA_PATH)/burst2 --seed=42
# 3 Hz
burst3:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=19.5 --n_cycles=3 --rate=3.0 --n=1000 --name=$(DATA_PATH)/burst3 --seed=42
# 4 Hz
burst4:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=19.5 --n_cycles=3 --rate=4.0 --n=1000 --name=$(DATA_PATH)/burst4 --seed=42
# 5 hz
burst5:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=19.5 --n_cycles=3 --rate=5.0 --n=1000 --name=$(DATA_PATH)/burst5 --seed=42
# 6 hz  (equal to stim == 'patho')
burst6:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=19.5 --n_cycles=3 --rate=6.0 --n=1000 --name=$(DATA_PATH)/burst6 --seed=42

# --------------------------------------------------------------
# 4-17-2018
# Intial testing, N=100 cells, 25 trials, t = 22, tau_h = 2, all 10 stim, all osc levels
# Ca: 0.031 (~ 6-12 Hz firing rate).
# Cut down osc sets to so there are only 12 jobs total
exp1:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp1.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp1_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=2 --burn_t=18 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 1 2 ::: 0 1 3 5 | tee $(DATA_PATH)/exp1.log

exp2:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp2.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp2_stim{1}_osc{2}_Ca{3} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=2 --burn_t=18 --Ca_target={3} --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 ::: 1 3 5 ::: 0.032 0.034 0.036 0.038 | tee $(DATA_PATH)/exp2.log