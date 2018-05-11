SHELL=/bin/bash -O expand_aliases
DATA_PATH=/Users/type/Code/resistingrhythm/data/
# DATA_PATH=/home/ejp/src/resistingrhythm/data
# DATA_PATH=/home/ubuntu/src/resistingrhythm/data

create_data_path:
	-mkdir $(DATA_PATH)

# --------------------------------------------------------------
all_stim: stim0 stim1 stim2 osc0 osc1 osc2 osc3 osc4 osc5 osc6 osc8 osc9 osc10 osc11 osc12 osc13 osc14 osc14 osc16 burst0 burst1 burst2 burst3 burst4 burst5 burst6 burst7 burst8 burst9 burst10 burst11 burst12 burst13 burst14 burst15 burst16 osc100 burst100 burst200 currents osc200 osc300 osc400 burst300 burst400 burst3000 burst500 stim3 stim4 osc1000


# Replicate 3 sitm, same params, different draws
stim0:
	python homeostasis_experiments.py create_stim --t=20 --t_onset=0 --w=20 --rate=6 --n=1000 --name=$(DATA_PATH)/stim0 --seed=23434
stim1:
	python homeostasis_experiments.py create_stim --t=20 --t_onset=0 --w=20 --rate=6 --n=1000 --name=$(DATA_PATH)/stim1 --seed=87
stim2:
	python homeostasis_experiments.py create_stim --t=20 --t_onset=0 --w=20 --rate=6 --n=1000 --name=$(DATA_PATH)/stim2 --seed=5318

# Stim comes on only at the end (19.5, assuming were working with a 8 Hz osc/burst)
# (Confirmed onset w/ vis)
stim3:
	python homeostasis_experiments.py create_stim --t=20 --t_onset=19.5 --w=20 --rate=6 --n=1000 --name=$(DATA_PATH)/stim3 --seed=23434
stim4:
	python homeostasis_experiments.py create_stim --t=20 --t_onset=19.5 --w=20 --rate=6 --n=1000 --name=$(DATA_PATH)/stim4 --seed=87

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
osc8:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=0 --n_cycles=180 --rate=0.05 --n=1000 --name=$(DATA_PATH)/osc8 --seed=42

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

osc3000:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=0 --n_cycles=180 --rate=30 --n=1000 --name=$(DATA_PATH)/osc3000 --seed=42


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

# Explore weak bursts 0.05 -> 0.5 Hz
burst7:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=19.5 --n_cycles=3 --rate=0.05 --n=1000 --name=$(DATA_PATH)/burst7 --seed=42

burst8:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=19.5 --n_cycles=3 --rate=0.1 --n=1000 --name=$(DATA_PATH)/burst8 --seed=42

burst9:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=19.5 --n_cycles=3 --rate=0.15 --n=1000 --name=$(DATA_PATH)/burst9 --seed=42

burst10:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=19.5 --n_cycles=3 --rate=0.2 --n=1000 --name=$(DATA_PATH)/burst10 --seed=42

burst11:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=19.5 --n_cycles=3 --rate=0.25 --n=1000 --name=$(DATA_PATH)/burst11 --seed=42

burst12:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=19.5 --n_cycles=3 --rate=0.3 --n=1000 --name=$(DATA_PATH)/burst12 --seed=42

burst13:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=19.5 --n_cycles=3 --rate=0.35 --n=1000 --name=$(DATA_PATH)/burst13 --seed=42

burst14:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=19.5 --n_cycles=3 --rate=0.4 --n=1000 --name=$(DATA_PATH)/burst14 --seed=42

burst15:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=19.5 --n_cycles=3 --rate=0.45 --n=1000 --name=$(DATA_PATH)/burst15 --seed=42

burst16:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=19.5 --n_cycles=3 --rate=0.45 --n=1000 --name=$(DATA_PATH)/burst16 --seed=42

burst3000:
	python homeostasis_experiments.py create_osc --t=20 --t_onset=19.5 --n_cycles=3 --rate=30 --n=1000 --name=$(DATA_PATH)/burst3000 --seed=42

# --------------------------------------------------------------
# 4-23-2018
# After trying a few different ranges / sampling schemes for osc_rate
# settled on a 0-0.5 by 0.1 Hz, then 0.5 - 6.0 by 0.5 Hz, dual approach.
# Run times sampling only at 0.1 are > days. Too slow/expensive.
osc100:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/osc100.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python homeostasis_experiments.py create_osc --t=20 --t_onset={3} --n_cycles={4} --rate={2} --n=1000 --name=$(DATA_PATH)/osc{1} --seed=42' :::: \
		osc100.txt

# -
# 4-29-2018
# f = 30
osc200:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/osc200.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python homeostasis_experiments.py create_osc --t=20 --t_onset={3} --n_cycles={4} --f={5} --rate={2} --n=1000 --name=$(DATA_PATH)/osc{1} --seed=42' :::: \
		osc200.txt

# f = 12
osc300:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/osc300.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python homeostasis_experiments.py create_osc --t=20 --t_onset={3} --n_cycles={4} --f={5} --rate={2} --n=1000 --name=$(DATA_PATH)/osc{1} --seed=42' :::: \
		osc300.txt

# f = 30
osc400:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/osc400.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python homeostasis_experiments.py create_osc --t=20 --t_onset={3} --n_cycles={4} --f={5} --rate={2} --n=1000 --name=$(DATA_PATH)/osc{1} --seed=42' :::: \
		osc400.txt

# -
# 5-1-2018
osc1000:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/osc1000.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python homeostasis_experiments.py create_osc --t=20 --t_onset={3} --n_cycles={4} --rate={2} --n=1000 --name=$(DATA_PATH)/osc{1} --seed=42' :::: \
		osc1000.txt


# ------------------------------------------------------------------------
# (Confirmed onset w/ vis)
burst100:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/burst100.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python homeostasis_experiments.py create_osc --t=20 --t_onset={3} --n_cycles={4} --rate={2} --n=1000 --name=$(DATA_PATH)/burst{1} --seed=42' :::: \
		burst100.txt

# f = 30
# (Confirmed onset w/ vis)
burst200:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/burst200.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python homeostasis_experiments.py create_osc --t=20 --t_onset={3} --n_cycles={4} --f={5} --rate={2} --n=1000 --name=$(DATA_PATH)/burst{1} --seed=42' :::: \
		burst200.txt

# -
# 4-28-2018
# f = 12
# (Confirmed onset w/ vis)
burst300:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/burst300.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python homeostasis_experiments.py create_osc --t=20 --t_onset={3} --n_cycles={4} --f={5} --rate={2} --n=1000 --name=$(DATA_PATH)/burst{1} --seed=42' :::: \
		burst300.txt

# f = 20
burst400:
# (Confirmed onset w/ vis)
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/burst400.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python homeostasis_experiments.py create_osc --t=20 --t_onset={3} --n_cycles={4} --f={5} --rate={2} --n=1000 --name=$(DATA_PATH)/burst{1} --seed=42' :::: \
		burst400.txt

# ------------------------------------
# 4-29-2018
# Burst w/ 4 cycles; f = 8
burst500:
# (Confirmed onset w/ vis)
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/burst500.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python homeostasis_experiments.py create_osc --t=20 --t_onset={3} --n_cycles={4} --rate={2} --n=1000 --name=$(DATA_PATH)/burst{1} --seed=42' :::: \
		burst500.txt



# --------------------------------------------------------------
# 4-23-2019
# Manipulate the tone, mimicing ACh action, using applied external currents

# (Confirmed onset w/ vis)
currents: tonic_current1 tonic_current2 tonic_current3 step_current0 step_current1 step_current2 step_current3


# Tonic
# (Confirmed onset of all w/ vis)
tonic_current1:
	python homeostasis_experiments.py create_current --t=20 --t_onset=0 --t_offset=20  --I=-0.5e-6 --name=$(DATA_PATH)/tonic_current1

tonic_current2:
	python homeostasis_experiments.py create_current --t=20 --t_onset=0 --t_offset=20  --I=-0.75e-6 --name=$(DATA_PATH)/tonic_current2

tonic_current3:
	python homeostasis_experiments.py create_current --t=20 --t_onset=0 --t_offset=20  --I=-1.0e-6 --name=$(DATA_PATH)/tonic_current3

# Burst
step_current0:
	python homeostasis_experiments.py create_current --t=20 --t_onset=19.85 --t_offset=20  --I=-0.0e-6 --name=$(DATA_PATH)/step_current0

step_current1:
	python homeostasis_experiments.py create_current --t=20 --t_onset=19.85 --t_offset=20  --I=-0.5e-6 --name=$(DATA_PATH)/step_current1

step_current2:
	python homeostasis_experiments.py create_current --t=20 --t_onset=19.85 --t_offset=20  --I=-0.75e-6 --name=$(DATA_PATH)/step_current2

step_current3:
	python homeostasis_experiments.py create_current --t=20 --t_onset=19.85 --t_offset=20  --I=-1.0e-6 --name=$(DATA_PATH)/step_current3

# --------------------------------------------------------------
#  Added test recipe 
test1:
	parallel -j 1 -v \
		--joblog '$(DATA_PATH)/test1.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/test1_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=18 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 ::: 1 | tee $(DATA_PATH)/test1.log

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
	'python homeostasis_experiments.py run $(DATA_PATH)/exp1_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=18 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
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
	'python homeostasis_experiments.py run $(DATA_PATH)/exp2_stim{1}_osc{2}_Ca{3} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=18 --Ca_target={3} --Ca_ref={3} --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 ::: 1 3 5 ::: 0.032 0.034 0.036 0.038 | tee $(DATA_PATH)/exp2.log

# Burst version of exp1; not the shorter collection window
exp3:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp3.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp3_stim{1}_burst{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/burst{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 1 2 ::: 0 1 3 5 | tee $(DATA_PATH)/exp3.log

# --------------------------------------------------------------
# 4-19-2018 
# Explore oscillations in the 0.05, 0.5 Hz range. Is error already maxed by 0.5 Hz, as in exp1?
# Exp1 showed no effect on error with osc_rate. 
exp4:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp4.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp4_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=18 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 ::: 6 7 8 9 10 11 12 13 14 15 16 | tee $(DATA_PATH)/exp4.log

# Run a negative control for exp1 i.e, w/ no osc
exp5:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp5.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp5_stim{1} --stim_name=$(DATA_PATH)/stim{1}.csv  --N=100 --t=20 --tau_h=4 --burn_t=18 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose ' \
	::: 0 1 2 | tee $(DATA_PATH)/exp5.log

# Run a negative control for exp1 i.e, w/ no homeo
exp6:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp6.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp6_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=18 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose --homeostasis=False' \
	::: 0 1 2 ::: 0 1 3 5 | tee $(DATA_PATH)/exp6.log

# Partial repeat exp2, but this time holding Ca_ref constant.
exp7:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp7.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp7_stim{1}_osc{2}_Ca{3} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=18 --Ca_target={3} --Ca_ref=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 ::: 1 3 5 ::: 0.032 0.034 0.036 | tee $(DATA_PATH)/exp7.logs

# --------------------------------------------------------------
# 4-20-2018
# Rerun weak osc, but with the same collection window as burst
exp8:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp8.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp8_stim{1}_burst{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 ::: 7 8 9 10 11 12 13 14 15 16 | tee $(DATA_PATH)/exp8.log

# TODO
# Weak osc, with no homeostasis (control)
exp9:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp9.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp9_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=18 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose --homeostasis=False' \
	::: 0 ::: 6 7 8 9 10 11 12 13 14 15 16 | tee $(DATA_PATH)/exp9.log


# --------------------------------------------------------------
# 4-21-2018
# -
# Run missing values for strong:
# osc
exp10:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp10.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp10_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=18 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 1 2 ::: 2 4 6 | tee $(DATA_PATH)/exp10.log

# burst
exp11:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp11.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp11_stim{1}_burst{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/burst{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 1 2 ::: 2 4 6 | tee $(DATA_PATH)/exp11.log

# -
# Rerun osc with a burst collection window
exp12:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp12.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp12_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 1 2 ::: 0 1 2 3 4 5 6 8 9 10 11 12 13 14 15 16 | tee $(DATA_PATH)/exp12.log

# -
# Fill in missing stim for weak osc
exp13:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp13.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp13_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=18 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 1 2 ::: 8 9 10 11 12 13 14 15 16 | tee $(DATA_PATH)/exp13.log

# -
# Re-run weak bursts (...repeating exp8 this time with actual bursts)
exp14:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp14.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp14_stim{1}_burst{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/burst{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 1 2 ::: 7 8 9 10 11 12 13 14 15 16 | tee $(DATA_PATH)/exp14.log

# --------------------------------------------------------------
# 4-23-2018
# - Begin FINAL exps -
# All use 2 stim, f = 8 (unless noted). 
# Analysis window is always 3+1 cycles:
# 19.5 for f = 8
# 19.85 for f = 30

# ---
# Run sustained oscillation with the `100` sampling scheme
exp15:
	parallel -j 36 -v \
		--joblog '$(DATA_PATH)/exp15.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp15_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 1 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp15.log

# Run burst with the 100 sampling scheme
exp16:
	parallel -j 36 -v \
		--joblog '$(DATA_PATH)/exp16.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp16_stim{1}_burst{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/burst{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 1 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp16.log

# ---
# Run f=30 burst, with a matched -step in ACh tone
exp17:
	parallel -j 48 -v \
		--joblog '$(DATA_PATH)/exp17.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp17_stim{1}_burst{2}_step{3} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/burst{2}.csv --current_name=$(DATA_PATH)/step_current{3}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.85 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 1 ::: 200 201 202 203 204 205 ::: 0 1 2 3 | tee $(DATA_PATH)/exp17.log


# Run sustained oscillations with tonic changes in ACh tone
exp18:
	parallel -j 48 -v \
		--joblog '$(DATA_PATH)/exp18.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp18_stim{1}_osc{2}_tonic{3} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --current_name=$(DATA_PATH)/tonic_current{3}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 1 ::: 101 103 105 115 125 135 145 155  ::: 1 2 3 | tee $(DATA_PATH)/exp18.log

# ---
# Run sustained oscillations, varying Ca_target:
# 0.026 0.028 0.032 0.034
exp19:
	parallel -j 64 -v \
		--joblog '$(DATA_PATH)/exp19.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp19_stim{1}_osc{2}_ca{3} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --w_min=5e-6 --w_max=50e-6 --Ca_target={3} --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 1 ::: 101 103 105 115 125 135 145 155 ::: 0.026 0.028 0.032 0.034 | tee $(DATA_PATH)/exp19.log

# Run bursts, varying the Ca_target:
# 0.026 0.028 0.032 0.034
exp20:
	parallel -j 48 -v \
		--joblog '$(DATA_PATH)/exp20.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp20_stim{1}_burst{2}_ca{3} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/burst{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --w_min=5e-6 --Ca_target={3} --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 1 ::: 101 102 103 104 105 110 ::: 0.026 0.028 0.032 0.034 | tee $(DATA_PATH)/exp20.log

# Run sustained oscillations, with V_osc set to mimic a GABA synapse:
# E_rev = -80e-3, tau=10 ms
exp21:
	parallel -j 36 -v \
		--joblog '$(DATA_PATH)/exp21.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp21_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --V_osc=-80e-3 --tau_osc=10e-3 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 1 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp21.log

# --------------------------------------------------------------
# 4-24-2018

# ---
# Run sustained oscillation with the `100` sampling scheme
# **with NO homeostasis**.
exp22:
	parallel -j 36 -v \
		--joblog '$(DATA_PATH)/exp22.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp22_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose --homeostasis=False' \
	::: 0 1 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp22.log

# Run sustained oscillations, with V_osc set to mimic a GABA synapse:
# E_rev = -80e-3, tau=10 ms
# **with NO homeostasis**
exp23:
	parallel -j 36 -v \
		--joblog '$(DATA_PATH)/exp23.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp23_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --V_osc=-80e-3 --tau_osc=10e-3 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose --homeostasis=False' \
	::: 0 1 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp23.log

# --------------------------------------------------------------
# 4-29-2018
# Sustained osc, f=30
exp24:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp24.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp24_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.85 --w_min=5e-6 --w_max=50e-6 --Ca_target=0.03 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 1 ::: 205 215 225 235 245 255 | tee $(DATA_PATH)/exp24.log

# Sustained osc, f=12
exp25:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp25.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp25_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.66 --w_min=5e-6 --w_max=50e-6 --Ca_target=0.03 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 1 ::: 305 315 325 335 345 355 | tee $(DATA_PATH)/exp25.log

# Sustained osc, f=20
exp26:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp26.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp26_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.80 --w_min=5e-6 --w_max=50e-6 --Ca_target=0.03 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 1 ::: 405 415 425 435 445 555 | tee $(DATA_PATH)/exp26.log

# -
# Burst, f=30
exp27:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp27.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp27_stim{1}_burst{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/burst{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.85 --w_min=5e-6 --w_max=50e-6 --Ca_target=0.03 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 1 ::: 205 215 225 235 245 255 | tee $(DATA_PATH)/exp27.log

# Burst, f=12
exp28:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp28.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp28_stim{1}_burst{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/burst{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.66 --w_min=5e-6 --w_max=50e-6 --Ca_target=0.03 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 1 ::: 305 315 325 335 345 355 | tee $(DATA_PATH)/exp28.log

# Burst, f=20
exp29:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp29.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp29_stim{1}_burst{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/burst{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.80 --w_min=5e-6 --w_max=50e-6 --Ca_target=0.03 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 1 ::: 405 415 425 435 445 555 | tee $(DATA_PATH)/exp29.log

# -
# I bursts
# Run sustained oscillations, with V_osc set to mimic a GABA synapse:
# E_rev = -80e-3, tau=10 ms
# (Expected these to be suppresive; they are not; 
# the stim HHH effect adds a AHP spike which drives an overall excitbility
# response; if you crank the `osc_rate` to around 30 suppression effects return
# as these swamp out the AHP rebound spikes; need to run an osc only no stim
# set of trials to see what I only for a while does...)
exp30:
	parallel -j 36 -v \
		--joblog '$(DATA_PATH)/exp30.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp30_stim{1}_burst{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/burst{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --V_osc=-80e-3 --tau_osc=10e-3 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 1 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp30.log

# --------------------------------------------------------------
# 4-29-2017
# Rerun I and E bursts with 4 cycles in place of 3
# with 3 there was a 1 cycle mismatch with the osc
# exps.... sigh.
# burst500....
# (Made the KCa dynamic as well; wanted to rule out it's staticness from causing 
# the AHP effect discussed in exp30; dynamic KCa changes nothing important; I ran
# manual control exps with GABA f = 8, osc_rate=30, and NO H, and these neurons
# do not normally have a AHP effect, as expected; if they did that would be a 
# parameterization problem.)

# E
exp31:
	parallel -j 36 -v \
		--joblog '$(DATA_PATH)/exp31.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp31_stim{1}_burst{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/burst{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 1 ::: 500 501 502 503 504 505 510 515 520 525 530 535 540 545 550 555 560 | tee $(DATA_PATH)/exp31.log

# I
exp32:
	parallel -j 36 -v \
		--joblog '$(DATA_PATH)/exp32.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp32_stim{1}_burst{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/burst{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --V_osc=-80e-3 --tau_osc=10e-3 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 1 ::: 500 501 502 503 504 505 510 515 520 525 530 535 540 545 550 555 560 | tee $(DATA_PATH)/exp32.log

# --------------------------------------------------------------
# 4-30-2018
# Sustained oscillation with stim off until 19.5
# E
exp33:
	parallel -j 36 -v \
		--joblog '$(DATA_PATH)/exp33.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp33_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp33.log

# I
exp34:
	parallel -j 36 -v \
		--joblog '$(DATA_PATH)/exp34.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp34_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --V_osc=-80e-3 --tau_osc=10e-3 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp34.log

# --------------------------------------------------------------
# 5-1-2018
# Re-run exp33+34 at high osc_rate
# E
exp35:
	parallel -j 36 -v \
		--joblog '$(DATA_PATH)/exp35.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp35_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 3 4 ::: 1070 1080 1090 1110 1120 | tee $(DATA_PATH)/exp35.log

# I
exp36:
	parallel -j 36 -v \
		--joblog '$(DATA_PATH)/exp36.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp36_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --V_osc=-80e-3 --tau_osc=10e-3 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 3 4 ::: 1070 1080 1090 1110 1120 | tee $(DATA_PATH)/exp36.log

# --------------------------------------------------------------
# 5-5-2018
# Strong sustained I, full osc 1000 series
exp37:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp37.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp37_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --V_osc=-80e-3 --tau_osc=10e-3 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 1 ::: 1065 1070 1075 1080 1085 1090 1095 1100 1105 1110 1115 1120 1125 1130 1135 1140 1145 1150 1155 1160 1165 1170 1175 1180 | tee $(DATA_PATH)/exp37.log

# Sustained E, f = 12, 20, 30
# f=30
exp38:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp38.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp38_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.85 --w_min=5e-6 --w_max=50e-6 --Ca_target=0.03 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 1 ::: 205 215 225 235 245 255 | tee $(DATA_PATH)/exp38.log

# f=12
exp39:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp38.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp39_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.66 --w_min=5e-6 --w_max=50e-6 --Ca_target=0.03 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 1 ::: 305 315 325 335 345 355 | tee $(DATA_PATH)/exp29.log

# f=20
exp40:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp40.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp40_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.80 --w_min=5e-6 --w_max=50e-6 --Ca_target=0.03 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 1 ::: 405 415 425 435 445 555 | tee $(DATA_PATH)/exp40.log

# 5-1-2018
# 1f50d94e168f3fcab14382e54bbd988b61cb4110
# (balanced branch)

# Strong sustained I, the upper part of the osc 1000 series
exp41:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp41.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp41_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --V_osc=-80e-3 --tau_osc=10e-3 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --noise_rate=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 1 ::: 1065 1070 1075 1080 1085 1090 1095 1100 1105 1110 1115 1120 1125 1130 1135 1140 1145 1150 1155 1160 1165 1170 1175 1180 | tee $(DATA_PATH)/exp41.log
