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

# Explore weak oscillations 0.05 -> 0.5 Hz
osc7:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=0 --n_cycles=180 --rate=0.05 --n=1000 --name=$(DATA_PATH)/osc7 --seed=42

osc8:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=0 --n_cycles=180 --rate=0.1 --n=1000 --name=$(DATA_PATH)/osc8 --seed=42

osc9:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=0 --n_cycles=180 --rate=0.1 --n=1000 --name=$(DATA_PATH)/osc9 --seed=42

osc10:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=0 --n_cycles=180 --rate=0.15 --n=1000 --name=$(DATA_PATH)/osc10 --seed=42

osc11:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=0 --n_cycles=180 --rate=0.2 --n=1000 --name=$(DATA_PATH)/osc11 --seed=42

osc12:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=0 --n_cycles=180 --rate=0.25 --n=1000 --name=$(DATA_PATH)/osc12 --seed=42

osc13:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=0 --n_cycles=180 --rate=0.3 --n=1000 --name=$(DATA_PATH)/osc13 --seed=42

osc14:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=0 --n_cycles=180 --rate=0.35 --n=1000 --name=$(DATA_PATH)/osc14 --seed=42

osc15:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=0 --n_cycles=180 --rate=0.4 --n=1000 --name=$(DATA_PATH)/osc15 --seed=42

osc16:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=0 --n_cycles=180 --rate=0.45 --n=1000 --name=$(DATA_PATH)/osc16 --seed=42

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
#
# RESULT: increases osc_rate increase coord; error is NOT effected (which is odd)
exp1:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp1.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp1_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=2 --burn_t=18 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 1 2 ::: 0 1 3 5 | tee $(DATA_PATH)/exp1.log

# --------------------------------------------------------------
# 4-18-2018
# Repeat most of exp1, trying small increases in Ca_target
#
# RESULT: Increasing Ca increases both error and var. Ca increases and osc_rate
# increases amplify each other. 
exp2:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp2.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp2_stim{1}_osc{2}_Ca{3} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=2 --burn_t=18 --Ca_target={3} --Ca_ref={3} --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 ::: 1 3 5 ::: 0.032 0.034 0.036 0.038 | tee $(DATA_PATH)/exp2.log

# Burst version of exp1; not the shorter collection window
exp3:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp3.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp3_stim{1}_burst{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/burst{2}.csv --N=100 --t=20 --tau_h=2 --burn_t=19.5 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 1 2 ::: 0 1 3 5 | tee $(DATA_PATH)/exp3.log

# --------------------------------------------------------------
# 4-19-2018 
# Explore oscillations in the 0.05, 0.5 Hz range. Is error already maxed by 0.5 Hz, as in exp1?
# Exp1 showed no effect on error with osc_rate. 
exp4:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp4.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp4_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=2 --burn_t=18 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 ::: 6 7 8 9 10 11 12 13 14 15 16 | tee $(DATA_PATH)/exp4.log

# --------------------------------------------------------------
# TODO:
# Run a negative control for exp1 i.e, w/ no osc
exp5:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp5.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp5_stim{1} --stim_name=$(DATA_PATH)/stim{1}.csv  --N=100 --t=20 --tau_h=2 --burn_t=18 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose ' \
	::: 0 1 2 | tee $(DATA_PATH)/exp5.log

# Run a negative control for exp1 i.e, w/ no homeo
exp6:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp6.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp6_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=2 --burn_t=18 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose --no_homeostasis' \
	::: 0 1 2 ::: 0 1 3 5 | tee $(DATA_PATH)/exp6.log

# Repeat exp2, but this time holding Ca_ref constant.
exp7:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp7.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp7_stim{1}_osc{2}_Ca{3} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=2 --burn_t=18 --Ca_target={3} --Ca_ref=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 ::: 1 3 5 ::: 0.032 0.034 0.036 | tee $(DATA_PATH)/exp7.logs