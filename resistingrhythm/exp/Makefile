SHELL=/bin/bash -O expand_aliases
# DATA_PATH=/Users/type/Code/resistingrhythm/data/
DATA_PATH=/home/ejp/src/resistingrhythm/data
# DATA_PATH=/home/ubuntu/src/resistingrhythm/data
# DATA_PATH=/home/stitch/Code/resistingrhythm/data
# DATA_PATH=/Users/qualia/Code/resistingrhythm/data

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
		'python homeostasis_experiments.py create_osc --t=20 --t_onset={3} --n_cycles={4} --rate={2} --min_rate=2.0 --n=1000 --name=$(DATA_PATH)/osc{1} --seed=42' :::: \
		osc100.txt

# -
# 4-29-2018
# f = 30
osc200:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/osc200.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python homeostasis_experiments.py create_osc --t=20 --t_onset={3} --n_cycles={4} --f={5} --rate={2} --min_rate=2.0 --n=1000 --name=$(DATA_PATH)/osc{1} --seed=42' :::: \
		osc200.txt

# f = 12
osc300:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/osc300.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python homeostasis_experiments.py create_osc --t=20 --t_onset={3} --n_cycles={4} --f={5} --rate={2} --min_rate=2.0 --n=1000 --name=$(DATA_PATH)/osc{1} --seed=42' :::: \
		osc300.txt

# f = 30
osc400:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/osc400.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python homeostasis_experiments.py create_osc --t=20 --t_onset={3} --n_cycles={4} --f={5} --rate={2} --min_rate=2.0 --n=1000 --name=$(DATA_PATH)/osc{1} --seed=42' :::: \
		osc400.txt

# -
# 5-1-2018
osc1000:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/osc1000.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python homeostasis_experiments.py create_osc --t=20 --t_onset={3} --n_cycles={4} --rate={2} --n=1000 --min_rate=2.0 --name=$(DATA_PATH)/osc{1} --seed=42' :::: \
		osc1000.txt


# ----------------------------------------------------------------------------
# 5-8-2019
# Generate hippo LFP oscillations
lfp100:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/lfp100.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python homeostasis_experiments.py create_hippocampus_theta --t=20 --t_onset={3} --n_cycles={4} --rate={2} --min_rate=2.0 --n=1000 --name=$(DATA_PATH)/lfp{1} --seed=42' :::: \
		osc100.txt

# ---------------------------------------------------------------------------
# 5-8-2019
# freq: 6, 7, 9, 10

oscsix100:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/osc100.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python homeostasis_experiments.py create_osc --t=20 --t_onset={3} --n_cycles={4} --rate={2} --min_rate=2.0 --n=1000 --f=6 --name=$(DATA_PATH)/oscsix{1} --seed=42' :::: \
		osc100.txt

oscseven100:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/osc100.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python homeostasis_experiments.py create_osc --t=20 --t_onset={3} --n_cycles={4} --rate={2} --min_rate=2.0 --n=1000 --f=7 --name=$(DATA_PATH)/oscseven{1} --seed=42' :::: \
		osc100.txt

oscnine100:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/osc100.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python homeostasis_experiments.py create_osc --t=20 --t_onset={3} --n_cycles={4} --rate={2} --min_rate=2.0 --n=1000 --f=9 --name=$(DATA_PATH)/oscnine{1} --seed=42' :::: \
		osc100.txt

oscten100:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/osc100.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python homeostasis_experiments.py create_osc --t=20 --t_onset={3} --n_cycles={4} --rate={2} --min_rate=2.0 --n=1000 --f=10 --name=$(DATA_PATH)/oscten{1} --seed=42' :::: \
		osc100.txt

# ---------------------------------------------------------------------------
# (Confirmed onset w/ vis)
burst100:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/burst100.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python homeostasis_experiments.py create_osc --t=20 --t_onset={3} --n_cycles={4} --rate={2} --n=1000 --min_rate=2.0 --name=$(DATA_PATH)/burst{1} --seed=42' :::: \
		burst100.txt

# ---------------------------------------------------------------------------
# Gamma bursts
# 5-8-2019

# f = 60
lowgamma100:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/lowgamma100.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python homeostasis_experiments.py create_osc --t=20 --t_onset={3} --n_cycles={4} --rate={2} --n=1000 --min_rate=2.0 --f=60 --name=$(DATA_PATH)/lowgamma{1} --seed=42' :::: \
		burst100.txt

# f = 90
highgamma100:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/highamma100.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python homeostasis_experiments.py create_osc --t=20 --t_onset={3} --n_cycles={4} --rate={2} --n=1000 --min_rate=2.0 --f=90 --name=$(DATA_PATH)/highgamma{1} --seed=42' :::: \
		burst100.txt

# ---------------------------------------------------------------------------
# Ripple bursts
# 5-8-2019

# f = 150
ripple100:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/ripple100.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python homeostasis_experiments.py create_osc --t=20 --t_onset={3} --n_cycles={4} --rate={2} --n=1000 --min_rate=2.0 --f=150 --name=$(DATA_PATH)/ripple{1} --seed=42' :::: \
		burst100.txt

# ---------------------------------------------------------------------------
# f = 30
# (Confirmed onset w/ vis)
burst200:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/burst200.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python homeostasis_experiments.py create_osc --t=20 --t_onset={3} --n_cycles={4} --f={5} --rate={2} --min_rate=2.0 --n=1000 --name=$(DATA_PATH)/burst{1} --seed=42' :::: \
		burst200.txt

# -
# 4-28-2018
# f = 12
# (Confirmed onset w/ vis)
burst300:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/burst300.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python homeostasis_experiments.py create_osc --t=20 --t_onset={3} --n_cycles={4} --f={5} --rate={2} --min_rate=2.0 --n=1000 --name=$(DATA_PATH)/burst{1} --seed=42' :::: \
		burst300.txt

# f = 20
burst400:
# (Confirmed onset w/ vis)
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/burst400.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python homeostasis_experiments.py create_osc --t=20 --t_onset={3} --n_cycles={4} --f={5} --rate={2} --min_rate=2.0 --n=1000 --name=$(DATA_PATH)/burst{1} --seed=42' :::: \
		burst400.txt

# ------------------------------------
# 4-29-2018
# Burst w/ 4 cycles; f = 8
burst500:
# (Confirmed onset w/ vis)
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/burst500.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python homeostasis_experiments.py create_osc --t=20 --t_onset={3} --n_cycles={4} --rate={2} --min_rate=2.0 --n=1000 --name=$(DATA_PATH)/burst{1} --seed=42' :::: \
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

# All use 2 stim, f = 8 (unless noted). 
# Analysis window is always 3+1 cycles:
# 19.5 for f = 8
# 19.85 for f = 30

# ---
# Run sustained oscillation with the `100` sampling scheme
exp15:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp15.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp15_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 1 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp15.log

# Run burst with the 100 sampling scheme
exp16:
	parallel -j 12 -v \
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
	parallel -j 12 -v \
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
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp22.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp22_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose --homeostasis=False' \
	::: 0 1 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp22.log

# Run sustained oscillations, with V_osc set to mimic a GABA synapse:
# E_rev = -80e-3, tau=10 ms
# **with NO homeostasis**
exp23:
	parallel -j 12 -v \
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
	parallel -j 12 -v \
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
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp31.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp31_stim{1}_burst{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/burst{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 1 ::: 500 501 502 503 504 505 510 515 520 525 530 535 540 545 550 555 560 | tee $(DATA_PATH)/exp31.log

# I
exp32:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp32.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp32_stim{1}_burst{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/burst{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --V_osc=-80e-3 --tau_osc=10e-3 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 0 1 ::: 500 501 502 503 504 505 510 515 520 525 530 535 540 545 550 555 560 | tee $(DATA_PATH)/exp32.log

# --------------------------------------------------------------
# 4-30-2018
# Sustained oscillation with stim off until 19.5
# E
exp33:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp33.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp33_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp33.log

# I
exp34:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp34.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp34_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --V_osc=-80e-3 --tau_osc=10e-3 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp34.log

# --------------------------------------------------------------
# 5-1-2018
# Re-run exp33+34 at high osc_rate
# E
exp35:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp35.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp35_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=25 --seed_value=42 --verbose' \
	::: 3 4 ::: 1070 1080 1090 1110 1120 | tee $(DATA_PATH)/exp35.log

# I
exp36:
	parallel -j 12 -v \
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

# ----------------------------------------------------------------------------
# 10-29-2018
# ec52b15e79e2ad3d72f20bf142e3f061b1edc4b5

# Run E and I oscillations w/
# - min_rate ~ 2 (this is the key change)
#
# - stim of 6 that comes on only after burn in by modification of 
# the osc100 series

# RESULTS: E results look as expected. H inversion, as before w/ stim on the 
# whole time. 

# Note: Realized in doing these analysese that the coord metric 'kappa' 
# time constant was WAY too small (1e-5). Created a re-run of these exps w/ 
# tau=1ms, as kappa is a better metric than var_mse.

# ----------------------------------------------------------------------------
# Sustained

# Excitatory
# no H
exp200:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp200.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp200_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=False' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp200.log

# H
exp201:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp201.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp201_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=True' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp201.log

# Inhibitory
# no H
exp202:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp202.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp202_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --V_osc=-80e-3 --tau_osc=10e-3 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=False' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp202.log

# H
exp203:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp203.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp203_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --V_osc=-80e-3 --tau_osc=10e-3 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=True' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp203.log


# ----------------------------------------------------------------------------
# Burst

# Excitatory
# H
exp204:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp204.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp204_stim{1}_burst{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/burst{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=True' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp204.log

# No H
exp205:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp205.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp205_stim{1}_burst{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/burst{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=False' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp205.log


# Inhibitory
# no H
exp206:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp206.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp206_stim{1}_burst{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/burst{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --V_osc=-80e-3 --tau_osc=10e-3 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=False' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp206.log

# H
exp207:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp207.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp207_stim{1}_burst{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/burst{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --V_osc=-80e-3 --tau_osc=10e-3 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=True' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp207.log

# ----------------------------------------------------------------------------
# 10-30-2018
# Re-run of exp200-207 w/ corrected kappa and Ca
# 3c8fd61dad477ce2868477cee3a916f914e79789

# ----------------------------------------------------------------------------
# Sustained

# Excitatory
# no H
exp210:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp210.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp210_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=False' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp210.log

# H
exp211:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp211.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp211_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=True' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp211.log

# Inhibitory
# no H
exp212:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp212.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp212_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --V_osc=-80e-3 --tau_osc=10e-3 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=False' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp212.log

# H
exp213:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp213.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp213_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --V_osc=-80e-3 --tau_osc=10e-3 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=True' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp213.log


# ----------------------------------------------------------------------------
# Burst

# Excitatory
# H
exp214:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp214.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp214_stim{1}_burst{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/burst{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=True' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp214.log

# No H
exp215:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp215.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp215_stim{1}_burst{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/burst{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=False' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp215.log


# Inhibitory
# H
exp216:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp216.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp216_stim{1}_burst{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/burst{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --V_osc=-80e-3 --tau_osc=10e-3 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=False' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp216.log

# no H
exp217:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp217.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp217_stim{1}_burst{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/burst{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --V_osc=-80e-3 --tau_osc=10e-3 --Ca_target=0.03 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=True' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp217.log

# ----------------------------------------------------------------------------
# 10-31-2018
# 3c8fd61dad477ce2868477cee3a916f914e79789
# Vary calcium, sustained E w/ H.
# 0.026 0.028 0.032 0.034
exp218:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp218.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp218_stim{1}_osc{2}_ca{3} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --w_min=5e-6 --w_max=50e-6 --Ca_target={3} --sigma=0 --num_trials=10 --seed_value=42 --verbose' \
	::: 3 ::: 101 103 105 115 125 135 145 155 ::: 0.026 0.028 0.032 0.034 | tee $(DATA_PATH)/exp218.log


# ----------------------------------------------------------------------------
# 11-9-2018
# Tweaked G/g intial values and re-balanced Na/K conductances. This removed
# The AHP spike seen in exp20* and exp21*

# 492af379e5f129ce70d6e78f27d50a76881da6fc

# ----------------------------------------------------------------------------
# Sustained

# Excitatory
# no H
exp220:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp220.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp220_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=False' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp220.log

# H
exp221:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp221.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp221_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=True' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp221.log

# Inhibitory
# no H
exp222:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp222.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp222_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --V_osc=-80e-3 --tau_osc=10e-3 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=False' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp222.log

# H
exp223:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp223.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp223_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --V_osc=-80e-3 --tau_osc=10e-3 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=True' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp223.log


# ----------------------------------------------------------------------------
# Burst

# Excitatory
# H
exp224:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp224.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp224_stim{1}_burst{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/burst{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=True' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp224.log

# No H
exp225:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp225.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp225_stim{1}_burst{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/burst{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=False' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp225.log


# Inhibitory
# H
exp226:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp226.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp226_stim{1}_burst{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/burst{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --V_osc=-80e-3 --tau_osc=10e-3 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=False' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp226.log

# no H
exp227:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp227.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp227_stim{1}_burst{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/burst{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --V_osc=-80e-3 --tau_osc=10e-3 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=True' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp227.log

# ----------------------------------------------------------------------------
# Vary calcium, sustained E w/ H.
# 0.026 0.028 0.032 0.034
exp228:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp228.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp228_stim{1}_osc{2}_ca{3} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --w_min=5e-6 --w_max=50e-6 --Ca_target={3} --sigma=0 --num_trials=10 --seed_value=42 --verbose' \
	::: 3 ::: 101 103 105 115 125 135 145 155 ::: 0.0026 0.0028 0.0032 0.0034 | tee $(DATA_PATH)/exp228.log

# ----------------------------------------------------------------------------
# 11-12-2018
# Do f and r_o matter. So some dose-response exps.
#
# tau_h ratio exps
# T = tau_h * (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
# T = (1, 2, 3, 4, 5, 6, 7, 8)
# f = (12, 16, 20, 24, 28)
# r = (0, 2, 4, 6)
exp229:
	# Osc files
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp229_osc.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py create_osc --t={1} --t_onset=0.0 --n_cycles=500 --rate={3} --f={4} --min_rate=2.0 --n=1000 --name=$(DATA_PATH)/osc_exp229_T{1}_r{3}_f{4} --seed=42' :::: exp229.txt
	# Stim files
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp229_stim.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py create_stim --t={1} --t_onset={2} --w=20 --rate=6 --n=1000 --name=$(DATA_PATH)/stim_exp229_t{2}_T{1} --seed=23434' :::: exp229.txt
	# Run exp
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp229.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp229_T{1}_t{2}_r{3}_f{4}.csv --stim_name=$(DATA_PATH)/stim_exp229_t{2}_T{1}.csv --osc_name=$(DATA_PATH)/osc_exp229_T{1}_r{3}_f{4}.csv --N=100 --t={1} --tau_h=4 --burn_t={2} --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=True' :::: exp229.txt

# 11-14-2018
# Repeat 229 w/out homeostasis
exp230:
	# Osc files
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp230_osc.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py create_osc --t={1} --t_onset=0.0 --n_cycles=500 --rate={3} --f={4} --min_rate=2.0 --n=1000 --name=$(DATA_PATH)/osc_exp230_T{1}_r{3}_f{4} --seed=42' :::: exp229.txt
	# Stim files
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp230_stim.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py create_stim --t={1} --t_onset={2} --w=20 --rate=6 --n=1000 --name=$(DATA_PATH)/stim_exp230_t{2}_T{1} --seed=23434' :::: exp229.txt
	# Run exp
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp230.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp230_T{1}_t{2}_r{3}_f{4}.csv --stim_name=$(DATA_PATH)/stim_exp230_t{2}_T{1}.csv --osc_name=$(DATA_PATH)/osc_exp230_T{1}_r{3}_f{4}.csv --N=100 --t={1} --tau_h=4 --burn_t={2} --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=False' :::: exp229.txt


# ----------------------------------------------------------------------------
# 11-14-2018
# Long tau_h: do bursts matter when tau_h is realistic? More dose-response exp
# 192bc077ad5f1f8b8129bf4661f114110f1d7cb5
exp231:
	# ------------------------------------------------------------------------
	# Stim/osc file generation
	# Osc:
	# 4 cycle
	python homeostasis_experiments.py create_osc --t=0.5 --t_onset=0 --f=8 --n_cycles=4 --min_rate=2 --rate=6 --n=1000 --name=$(DATA_PATH)/burst_exp231_c4 --seed=42
	# 8 cycle
	python homeostasis_experiments.py create_osc --t=1.0 --t_onset=0 --f=8 --n_cycles=8 --min_rate=2 --rate=6 --n=1000 --name=$(DATA_PATH)/burst_exp231_c8 --seed=42
	# 480 cycles (tonic: 1 min)
	python homeostasis_experiments.py create_osc --t=60.0 --t_onset=0 --f=8 --n_cycles=480 --min_rate=2 --rate=6 --n=1000 --name=$(DATA_PATH)/osc_exp231_c480 --seed=42
	# Stim:
	# 4 cycles
	python homeostasis_experiments.py create_stim --t=0.5 --t_onset=0.0 --w=20 --rate=6 --n=1000 --name=$(DATA_PATH)/stim_exp231_c4 --seed=23434
	# 4 cycles
	python homeostasis_experiments.py create_stim --t=1.0 --t_onset=0.5 --w=20 --rate=6 --n=1000 --name=$(DATA_PATH)/stim_exp231_c8 --seed=23434
	# 480 cycles
	python homeostasis_experiments.py create_stim --t=60.0 --t_onset=59.5 --w=20 --rate=6 --n=1000 --name=$(DATA_PATH)/stim_exp231_c480 --seed=23434
	# ------------------------------------------------------------------------
	# Exp
	# 4 cycle
	python homeostasis_experiments.py run $(DATA_PATH)/exp231_c4.csv --stim_name=$(DATA_PATH)/stim_exp231_c4.csv --osc_name=$(DATA_PATH)/burst_exp231_c4.csv --N=100 --t=0.5 --tau_h=600 --burn_t=0.0 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=20 --seed_value=42 --verbose --homeostasis=True
	# 8 cycle
	python homeostasis_experiments.py run $(DATA_PATH)/exp231_c8.csv --stim_name=$(DATA_PATH)/stim_exp231_c8.csv --osc_name=$(DATA_PATH)/burst_exp231_c8.csv --N=100 --t=1.0 --tau_h=600 --burn_t=0.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=20 --seed_value=42 --verbose --homeostasis=True
	# 480 cycle
	python homeostasis_experiments.py run $(DATA_PATH)/exp231_c480.csv --stim_name=$(DATA_PATH)/stim_exp231_c480.csv --osc_name=$(DATA_PATH)/osc_exp231_c480.csv --N=100 --t=60 --tau_h=600 --burn_t=59.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=20 --seed_value=42 --verbose --homeostasis=True

# Re-run exp231 w/ no H (this exp requires files created in 231).
exp232:
	# Exp
	# 4 cycle
	python homeostasis_experiments.py run $(DATA_PATH)/exp232_c4.csv --stim_name=$(DATA_PATH)/stim_exp231_c4.csv --osc_name=$(DATA_PATH)/burst_exp231_c4.csv --N=100 --t=0.5 --tau_h=600 --burn_t=0.0 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=20 --seed_value=42 --verbose --homeostasis=False
	# 8 cycle
	python homeostasis_experiments.py run $(DATA_PATH)/exp232_c8.csv --stim_name=$(DATA_PATH)/stim_exp231_c8.csv --osc_name=$(DATA_PATH)/burst_exp231_c8.csv --N=100 --t=1.0 --tau_h=600 --burn_t=0.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=20 --seed_value=42 --verbose --homeostasis=False
	# 480 cycle
	python homeostasis_experiments.py run $(DATA_PATH)/exp232_c480.csv --stim_name=$(DATA_PATH)/stim_exp231_c480.csv --osc_name=$(DATA_PATH)/osc_exp231_c480.csv --N=100 --t=60 --tau_h=600 --burn_t=59.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=20 --seed_value=42 --verbose --homeostasis=False


# ----------------------------------------------------------------------------
# Rerun freq dependence using a realistic tau_h
exp233:
	# Stim/osc file generation
	# Osc:
	# 4 cycle
	python homeostasis_experiments.py create_osc --t=0.5 --t_onset=0 --f=12 --n_cycles=4 --min_rate=2 --rate=6 --n=1000 --name=$(DATA_PATH)/burst_exp233_c4_f12 --seed=42
	python homeostasis_experiments.py create_osc --t=0.5 --t_onset=0 --f=20 --n_cycles=4 --min_rate=2 --rate=6 --n=1000 --name=$(DATA_PATH)/burst_exp233_c4_f20 --seed=42
	python homeostasis_experiments.py create_osc --t=0.5 --t_onset=0 --f=28 --n_cycles=4 --min_rate=2 --rate=6 --n=1000 --name=$(DATA_PATH)/burst_exp233_c4_f28 --seed=42
	# 8 cycle
	python homeostasis_experiments.py create_osc --t=1.0 --t_onset=0 --f=12 --n_cycles=8 --min_rate=2 --rate=6 --n=1000 --name=$(DATA_PATH)/burst_exp233_c8_f12 --seed=42
	python homeostasis_experiments.py create_osc --t=1.0 --t_onset=0 --f=20 --n_cycles=8 --min_rate=2 --rate=6 --n=1000 --name=$(DATA_PATH)/burst_exp233_c8_f20 --seed=42
	python homeostasis_experiments.py create_osc --t=1.0 --t_onset=0 --f=28 --n_cycles=8 --min_rate=2 --rate=6 --n=1000 --name=$(DATA_PATH)/burst_exp233_c8_f28 --seed=42
	# 480 cycles (tonic: 1 min)
	python homeostasis_experiments.py create_osc --t=60.0 --t_onset=0 --f=12 --n_cycles=480 --min_rate=2 --rate=6 --n=1000 --name=$(DATA_PATH)/osc_exp233_c480_f12 --seed=42
	python homeostasis_experiments.py create_osc --t=60.0 --t_onset=0 --f=20 --n_cycles=480 --min_rate=2 --rate=6 --n=1000 --name=$(DATA_PATH)/osc_exp233_c480_f20 --seed=42
	python homeostasis_experiments.py create_osc --t=60.0 --t_onset=0 --f=28 --n_cycles=480 --min_rate=2 --rate=6 --n=1000 --name=$(DATA_PATH)/osc_exp233_c480_f28 --seed=42
	# Stim:
	# 4 cycles
	python homeostasis_experiments.py create_stim --t=0.5 --t_onset=0.0 --w=20 --rate=6 --n=1000 --name=$(DATA_PATH)/stim_exp233_c4 --seed=23434
	# 4 cycles
	python homeostasis_experiments.py create_stim --t=1.0 --t_onset=0.5 --w=20 --rate=6 --n=1000 --name=$(DATA_PATH)/stim_exp233_c8 --seed=23434
	# 480 cycles
	python homeostasis_experiments.py create_stim --t=60.0 --t_onset=59.5 --w=20 --rate=6 --n=1000 --name=$(DATA_PATH)/stim_exp233_c480 --seed=23434
	# ------------------------------------------------------------------------
	# Exp
	# 4 cycle
	parallel -j 3 -v \
		--joblog '$(DATA_PATH)/exp233_c4.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python homeostasis_experiments.py run $(DATA_PATH)/exp233_c4_f{} --stim_name=$(DATA_PATH)/stim_exp233_c4.csv --osc_name=$(DATA_PATH)/burst_exp233_c4_f{}.csv --N=100 --t=0.5 --tau_h=600 --burn_t=0.0 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=20 --seed_value=42 --verbose --homeostasis=True' ::: 12 20 28
	# 8 cycle
	parallel -j 3 -v \
		--joblog '$(DATA_PATH)/exp233_c8.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python homeostasis_experiments.py run $(DATA_PATH)/exp233_c8_f{} --stim_name=$(DATA_PATH)/stim_exp233_c8.csv --osc_name=$(DATA_PATH)/burst_exp233_c8_f{}.csv --N=100 --t=1.0 --tau_h=600 --burn_t=0.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=20 --seed_value=42 --verbose --homeostasis=True' ::: 12 20 28
	# 480 cycle
	parallel -j 3 -v \
		--joblog '$(DATA_PATH)/exp233_c480.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python homeostasis_experiments.py run $(DATA_PATH)/exp233_c480_f{} --stim_name=$(DATA_PATH)/stim_exp233_c480.csv --osc_name=$(DATA_PATH)/osc_exp233_c480_f{}.csv --N=100 --t=60 --tau_h=600 --burn_t=59.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=20 --seed_value=42 --verbose --homeostasis=True' ::: 12 20 28


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# For the 220-230 series the dg_CA/dt term was commented out,
# and these were run w/ fixed Ca conductance. This is not correct?
# Rerun select.
# b8a0e310a6ba0ea188b233fc6b6b28549687bafc

# ----------------------------------------------------------------------------
# Excitatory
# no H
exp234:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp234.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp234_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=False' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp220.log

# H
exp235:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp235.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp235_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=True' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp235.log

# Inhibitory
# no H
exp236:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp236.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp236_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --V_osc=-80e-3 --tau_osc=10e-3 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=False' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp236.log

# H
exp237:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp237.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp237_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --V_osc=-80e-3 --tau_osc=10e-3 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=True' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp237.log


# ----------------------------------------------------------------------------
# Excitatory burst w H
exp238:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp238.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp238_stim{1}_burst{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/burst{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=True' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp238.log

# Inhibitory burst w H
exp239:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp239.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp239_stim{1}_burst{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/burst{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --V_osc=-80e-3 --tau_osc=10e-3 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=True' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 135 140 145 150 155 160 | tee $(DATA_PATH)/exp239.log


# ----------------------------------------------------------------------------
# Vary calcium, sustained E w/ H.
# 0.026 0.028 0.032 0.034
exp240:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp240.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp240_stim{1}_osc{2}_ca{3} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --w_min=5e-6 --w_max=50e-6 --Ca_target={3} --sigma=0 --num_trials=10 --seed_value=42 --verbose' \
	::: 3 ::: 101 103 105 115 125 135 145 155 ::: 0.0026 0.0028 0.0032 0.0034 | tee $(DATA_PATH)/exp240.log

# ----------------------------------------------------------------------------
# Realistic tau_h
exp241:
	# ------------------------------------------------------------------------
	# Stim/osc file generation
	# Osc:
	# 4 cycle
	python homeostasis_experiments.py create_osc --t=0.5 --t_onset=0 --f=8 --n_cycles=4 --min_rate=2 --rate=6 --n=1000 --name=$(DATA_PATH)/burst_exp241_c4 --seed=42
	# 8 cycle
	python homeostasis_experiments.py create_osc --t=1.0 --t_onset=0 --f=8 --n_cycles=8 --min_rate=2 --rate=6 --n=1000 --name=$(DATA_PATH)/burst_exp241_c8 --seed=42
	# 480 cycles (tonic: 1 min)
	python homeostasis_experiments.py create_osc --t=60.0 --t_onset=0 --f=8 --n_cycles=480 --min_rate=2 --rate=6 --n=1000 --name=$(DATA_PATH)/osc_exp241_c480 --seed=42
	# Stim:
	# 4 cycles
	python homeostasis_experiments.py create_stim --t=0.5 --t_onset=0.0 --w=20 --rate=6 --n=1000 --name=$(DATA_PATH)/stim_exp241_c4 --seed=23434
	# 4 cycles
	python homeostasis_experiments.py create_stim --t=1.0 --t_onset=0.5 --w=20 --rate=6 --n=1000 --name=$(DATA_PATH)/stim_exp241_c8 --seed=23434
	# 480 cycles
	python homeostasis_experiments.py create_stim --t=60.0 --t_onset=59.5 --w=20 --rate=6 --n=1000 --name=$(DATA_PATH)/stim_exp241_c480 --seed=23434
	# ------------------------------------------------------------------------
	# Exp
	# 4 cycle
	python homeostasis_experiments.py run $(DATA_PATH)/exp241_c4.csv --stim_name=$(DATA_PATH)/stim_exp241_c4.csv --osc_name=$(DATA_PATH)/burst_exp241_c4.csv --N=100 --t=0.5 --tau_h=600 --burn_t=0.0 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=20 --seed_value=42 --verbose --homeostasis=True
	# 8 cycle
	python homeostasis_experiments.py run $(DATA_PATH)/exp241_c8.csv --stim_name=$(DATA_PATH)/stim_exp241_c8.csv --osc_name=$(DATA_PATH)/burst_exp241_c8.csv --N=100 --t=1.0 --tau_h=600 --burn_t=0.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=20 --seed_value=42 --verbose --homeostasis=True
	# 480 cycle
	python homeostasis_experiments.py run $(DATA_PATH)/exp241_c480.csv --stim_name=$(DATA_PATH)/stim_exp241_c480.csv --osc_name=$(DATA_PATH)/osc_exp241_c480.csv --N=100 --t=60 --tau_h=600 --burn_t=59.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=20 --seed_value=42 --verbose --homeostasis=True

# Re-run exp231 w/ no H (this exp requires files created in 231).
exp242:
	# Exp
	# 4 cycle
	python homeostasis_experiments.py run $(DATA_PATH)/exp242_c4.csv --stim_name=$(DATA_PATH)/stim_exp241_c4.csv --osc_name=$(DATA_PATH)/burst_exp241_c4.csv --N=100 --t=0.5 --tau_h=600 --burn_t=0.0 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=20 --seed_value=42 --verbose --homeostasis=False
	# 8 cycle
	python homeostasis_experiments.py run $(DATA_PATH)/exp242_c8.csv --stim_name=$(DATA_PATH)/stim_exp241_c8.csv --osc_name=$(DATA_PATH)/burst_exp241_c8.csv --N=100 --t=1.0 --tau_h=600 --burn_t=0.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=20 --seed_value=42 --verbose --homeostasis=False
	# 480 cycle
	python homeostasis_experiments.py run $(DATA_PATH)/exp242_c480.csv --stim_name=$(DATA_PATH)/stim_exp241_c480.csv --osc_name=$(DATA_PATH)/osc_exp241_c480.csv --N=100 --t=60 --tau_h=600 --burn_t=59.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=20 --seed_value=42 --verbose --homeostasis=False


# ----------------------------------------------------------------------------
# Rerun freq dependence using a realistic tau_h
exp243:
	# Stim/osc file generation
	# Osc:
	# 4 cycle
	python homeostasis_experiments.py create_osc --t=0.5 --t_onset=0 --f=12 --n_cycles=4 --min_rate=2 --rate=6 --n=1000 --name=$(DATA_PATH)/burst_exp243_c4_f12 --seed=42
	python homeostasis_experiments.py create_osc --t=0.5 --t_onset=0 --f=20 --n_cycles=4 --min_rate=2 --rate=6 --n=1000 --name=$(DATA_PATH)/burst_exp243_c4_f20 --seed=42
	python homeostasis_experiments.py create_osc --t=0.5 --t_onset=0 --f=28 --n_cycles=4 --min_rate=2 --rate=6 --n=1000 --name=$(DATA_PATH)/burst_exp243_c4_f28 --seed=42
	# 8 cycle
	python homeostasis_experiments.py create_osc --t=1.0 --t_onset=0 --f=12 --n_cycles=8 --min_rate=2 --rate=6 --n=1000 --name=$(DATA_PATH)/burst_exp243_c8_f12 --seed=42
	python homeostasis_experiments.py create_osc --t=1.0 --t_onset=0 --f=20 --n_cycles=8 --min_rate=2 --rate=6 --n=1000 --name=$(DATA_PATH)/burst_exp243_c8_f20 --seed=42
	python homeostasis_experiments.py create_osc --t=1.0 --t_onset=0 --f=28 --n_cycles=8 --min_rate=2 --rate=6 --n=1000 --name=$(DATA_PATH)/burst_exp243_c8_f28 --seed=42
	# 480 cycles (tonic: 1 min)
	python homeostasis_experiments.py create_osc --t=60.0 --t_onset=0 --f=12 --n_cycles=480 --min_rate=2 --rate=6 --n=1000 --name=$(DATA_PATH)/osc_exp243_c480_f12 --seed=42
	python homeostasis_experiments.py create_osc --t=60.0 --t_onset=0 --f=20 --n_cycles=480 --min_rate=2 --rate=6 --n=1000 --name=$(DATA_PATH)/osc_exp243_c480_f20 --seed=42
	python homeostasis_experiments.py create_osc --t=60.0 --t_onset=0 --f=28 --n_cycles=480 --min_rate=2 --rate=6 --n=1000 --name=$(DATA_PATH)/osc_exp243_c480_f28 --seed=42
	# Stim:
	# 4 cycles
	python homeostasis_experiments.py create_stim --t=0.5 --t_onset=0.0 --w=20 --rate=6 --n=1000 --name=$(DATA_PATH)/stim_exp243_c4 --seed=23434
	# 4 cycles
	python homeostasis_experiments.py create_stim --t=1.0 --t_onset=0.5 --w=20 --rate=6 --n=1000 --name=$(DATA_PATH)/stim_exp243_c8 --seed=23434
	# 480 cycles
	python homeostasis_experiments.py create_stim --t=60.0 --t_onset=59.5 --w=20 --rate=6 --n=1000 --name=$(DATA_PATH)/stim_exp243_c480 --seed=23434
	# ------------------------------------------------------------------------
	# Exp
	# 4 cycle
	parallel -j 3 -v \
		--joblog '$(DATA_PATH)/exp243_c4.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python homeostasis_experiments.py run $(DATA_PATH)/exp243_c4_f{} --stim_name=$(DATA_PATH)/stim_exp243_c4.csv --osc_name=$(DATA_PATH)/burst_exp243_c4_f{}.csv --N=100 --t=0.5 --tau_h=600 --burn_t=0.0 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=20 --seed_value=42 --verbose --homeostasis=True' ::: 12 20 28
	# 8 cycle
	parallel -j 3 -v \
		--joblog '$(DATA_PATH)/exp243_c8.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python homeostasis_experiments.py run $(DATA_PATH)/exp243_c8_f{} --stim_name=$(DATA_PATH)/stim_exp243_c8.csv --osc_name=$(DATA_PATH)/burst_exp243_c8_f{}.csv --N=100 --t=1.0 --tau_h=600 --burn_t=0.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=20 --seed_value=42 --verbose --homeostasis=True' ::: 12 20 28
	# 480 cycle
	parallel -j 3 -v \
		--joblog '$(DATA_PATH)/exp243_c480.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
		'python homeostasis_experiments.py run $(DATA_PATH)/exp243_c480_f{} --stim_name=$(DATA_PATH)/stim_exp243_c480.csv --osc_name=$(DATA_PATH)/osc_exp243_c480_f{}.csv --N=100 --t=60 --tau_h=600 --burn_t=59.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=20 --seed_value=42 --verbose --homeostasis=True' ::: 12 20 28


# ----------------------------------------------------------------------------
# 5-8-2019
# Exps for reviewer response for JNP rapid review. In prep for re-submission
# as a reg. research report.

# 8e20b4ba3a8d5dff8ef143e9a3928571604c9cec

# ----------------------------------------------------------------------------
# 5-8-2019
# shape control. osc_rate: 100 101 102 103 104 105 110 115 120 125 130
# stim 3,4

# SUM:
# - H cond is not the what is expected based on exp220/221. 
# - See 278/279 for debugging these of reults.

# Excitatory
# no H
exp244:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp244.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp244_stim{1}lfp{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/lfp{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=False' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 | tee $(DATA_PATH)/exp220.log

# H
exp245:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp245.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp245_stim{1}_lfp{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/lfp{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=True' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 | tee $(DATA_PATH)/exp245.log

# Inhibitory
# no H
exp246:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp246.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp246_stim{1}_lfp{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/lfp{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --V_osc=-80e-3 --tau_osc=10e-3 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=False' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 | tee $(DATA_PATH)/exp246.log

# H
exp247:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp247.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp247_stim{1}_lfp{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/lfp{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --V_osc=-80e-3 --tau_osc=10e-3 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=True' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 | tee $(DATA_PATH)/exp247.log


# ---------------------------------------------------------------------------
# 5-8-2019

# SUM:
# - Things look like expected. 
# For bursts w/ and wout/ H, freq doesn't change the trend, but does change the
# overall strength of kappa and rate, as expected.

# 60 Hz gamma
# Excitatory burst w H
exp248:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp248.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp248_stim{1}_burst{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/lowgamma{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=True' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 | tee $(DATA_PATH)/exp248.log

# No H
exp249:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp249.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp249_stim{1}_burst{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/lowgamma{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=False' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 | tee $(DATA_PATH)/exp249.log

# ---------------------------------------------------------------------------
# 5-8-2019
# 90 Hz gmma
# Excitatory burst w H
exp250:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp250.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp250_stim{1}_burst{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/highgamma{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=True' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 | tee $(DATA_PATH)/exp250.log

# No H
exp251:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp251.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp251_stim{1}_burst{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/highgamma{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=False' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 | tee $(DATA_PATH)/exp251.log

# ----------------------------------------------------------------------------
# 5-8-2019
# 150 Hz ripple
# Excitatory burst w H
exp252:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp252.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp252_stim{1}_burst{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/ripple{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=True' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 | tee $(DATA_PATH)/exp252.log

# No H
exp253:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp253.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp253_stim{1}_burst{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/ripple{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=False' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 | tee $(DATA_PATH)/exp253.log


# ----------------------------------------------------------------------------
# 5-8-2019
# freq: 6,7,9,10

#  SUM:
# - H cond is not the what is expected based on exp220/221. 
# - See 278/279 for debugging these of reults.
# - Putting aside the H effect, f=6-9 look self-consistent. f=10 is quite 
#   different than all other exp conditions?
# - Given H issue all this will have to be re-run. Just focus on 7 and 9?
#   Faster run. Makes the point?

# Excitatory
# no H
exp254:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp254.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp254_stim{1}lfp{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/oscsix{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=False' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 | tee $(DATA_PATH)/exp220.log

# H
exp255:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp255.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp255_stim{1}_lfp{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/oscsix{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=True' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 | tee $(DATA_PATH)/exp255.log

# Inhibitory
# no H
exp256:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp256.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp256_stim{1}_lfp{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/oscsix{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --V_osc=-80e-3 --tau_osc=10e-3 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=False' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 | tee $(DATA_PATH)/exp256.log

# H
exp257:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp257.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp257_stim{1}_lfp{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/oscsix{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --V_osc=-80e-3 --tau_osc=10e-3 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=True' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 | tee $(DATA_PATH)/exp257.log

# --
# 7
# --

# Excitatory
# no H
exp258:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp258.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp258_stim{1}lfp{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/oscseven{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=False' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 | tee $(DATA_PATH)/exp258.log

# H
exp259:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp259.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp259_stim{1}_lfp{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/oscseven{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=True' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 | tee $(DATA_PATH)/exp259.log

# Inhibitory
# no H
exp260:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp260.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp260_stim{1}_lfp{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/oscseven{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --V_osc=-80e-3 --tau_osc=10e-3 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=False' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 | tee $(DATA_PATH)/exp260.log

# H
exp261:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp261.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp261_stim{1}_lfp{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/oscseven{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --V_osc=-80e-3 --tau_osc=10e-3 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=True' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 | tee $(DATA_PATH)/exp261.log

# --
# 9
# --

# Excitatory
# no H
exp262:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp262.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp262_stim{1}lfp{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/oscnine{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=False' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 | tee $(DATA_PATH)/exp262.log

# H
exp263:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp263.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp263_stim{1}_lfp{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/oscnine{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=True' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 | tee $(DATA_PATH)/exp263.log

# Inhibitory
# no H
exp264:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp264.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp264_stim{1}_lfp{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/oscnine{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --V_osc=-80e-3 --tau_osc=10e-3 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=False' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 | tee $(DATA_PATH)/exp265.log

# H
exp265:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp265.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp265_stim{1}_lfp{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/oscnine{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --V_osc=-80e-3 --tau_osc=10e-3 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=True' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 | tee $(DATA_PATH)/exp265.log

# --
# 10
# --

# Excitatory
# no H
exp266:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp266.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp266_stim{1}lfp{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/oscten{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=False' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 | tee $(DATA_PATH)/exp266.log

# H
exp267:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp267.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp267_stim{1}_lfp{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/oscten{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=True' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 | tee $(DATA_PATH)/exp267.log

# Inhibitory
# no H
exp268:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp268.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp268_stim{1}_lfp{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/oscten{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --V_osc=-80e-3 --tau_osc=10e-3 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=False' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 | tee $(DATA_PATH)/exp268.log

# H
exp269:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp269.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp269_stim{1}_lfp{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/oscten{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --V_osc=-80e-3 --tau_osc=10e-3 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=True' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 | tee $(DATA_PATH)/exp269.log


# ---------------------------------------------------------------------------
# 5-8-2019
# sigma: 360e-9 380e-9 400e-9 420e-9 440e-9

# SUM: 5-15-19 these runs were producing Nans. Killed them. Up dt to handle the 
# noise-driven numerical instability? Or just drop? Don't really need them for 
# rev. response

# Excitatory
# no H
exp270:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp270.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp270_stim{1}_sigma{3}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma={3} --num_trials=10 --seed_value=42 --verbose --homeostasis=False' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 ::: 360e-9 380e-9 400e-9 420e-9 440e-9 | tee $(DATA_PATH)/exp270.log

# H
exp271:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp271.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp271_stim{1}_sigma{3}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma={3} --num_trials=10 --seed_value=42 --verbose --homeostasis=True' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 ::: 360e-9 380e-9 400e-9 420e-9 440e-9 | tee $(DATA_PATH)/exp271.log


# ---------------------------------------------------------------------------
# 5-8-2019
# g_L: 0.9e-3, 0.95e-3, 1.05e-3, 1.1e-3

# SUM: 
# - Results for all conds are the _same_. 
# - Is a param not getting passed through?
# - H cond is not the what is expected based on exp220/221. 
# - See 278/279 for debugging these of reults.

# Excitatory
# no H
exp272:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp272.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp272_stim{1}_gl{3}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --g_l={3} --num_trials=10 --seed_value=42 --verbose --homeostasis=False' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 ::: 0.9e-3 0.95e-3 1.05e-3 1.1e-3 | tee $(DATA_PATH)/exp272.log

# H
exp273:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp273.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp273_stim{1}_gl{3}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --g_l={3} --num_trials=10 --seed_value=42 --verbose --homeostasis=True' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 ::: 0.9e-3 0.95e-3 1.05e-3 1.1e-3| tee $(DATA_PATH)/exp273.log


# ---------------------------------------------------------------------------
# 5-8-2019
# g_Ca: 0.027e-3, 0.028e-3, 0.029e-3, 0.031e-3, 0.032e-3, 0.033e-3

# SUM: 
# - Results for all conds are the _same_. 
# - Is a param not getting passed through?
# - H cond is not the what is expected based on exp220/221. 
# - See 278/279 for debugging these of reults.

# Excitatory
# no H
exp274:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp274.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp274_stim{1}_gCa{3}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --g_Ca={3} --num_trials=10 --seed_value=42 --verbose --homeostasis=False' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 ::: 0.027e-3 0.028e-3 0.029e-3 0.031e-3 0.032e-3 0.033e-3 | tee $(DATA_PATH)/exp274.log

# H
exp275:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp275.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp275_stim{1}_gCa{3}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --g_Ca={3} --num_trials=10 --seed_value=42 --verbose --homeostasis=True' \
	::: 3 4 ::: 100 101 102 103 104 105 110 115 120 125 130 ::: 0.027e-3 0.028e-3 0.029e-3 0.031e-3 0.032e-3 0.033e-3 | tee $(DATA_PATH)/exp275.log


# ----------------------------------------------------------------------------
# Exp244/45 (LFP exp) was quite different from the eq. sin wave version.
# Re-run the sin verion as a control; There have been several code changes
# since this it was last run.
#
# That is, direct repeat of exp220 and exp221. The std for H+AMPA result.
#
# SUM: Hell. 
#
# These are off compared to 220/221! 
#
# Something dramatic has changed in the code or stim generation. 
# Rather than suppressing activity, H is just slightly diminishing rate/kappa. 
# Either something is wrong, or there is parameter sensitivity I don't yet 
# understand.

# Excitatory
# no H
exp278:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp278.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp278_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=False' \
	::: 3 4 ::: 100 101 105 110 115 120 125 130 | tee $(DATA_PATH)/exp278.log

# H
exp279:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp279.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp279_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=True' \
	::: 3 4 ::: 100 101 105 110 115 120 125 130 | tee $(DATA_PATH)/exp279.log


# ----------------------------------------------------------------------------
# Debuging incogruent results.

# git branch rollback
# The rollback to 220-228
# git checkout 1a358a7aeb2b555062b00febe750279102a07705 
# 
# Re-run stim/osc:
# make osc100
# make stim3 stim4
# Finally, paste in and run the below

# SUM: 280/281 does return the expected effects. Hmm. Large g_CA does swamp 
# out strong H effects on action potential intiation. This makes sense. The emp.
# question is how strong is any Ca corrective currents in real neurons?
# - Need to do a senstivity sweep. 
# - Present w/out. Then sweep and show how it effects things.
# - Doesn't rule our H effect at all. Of course any H effect is directly 
#   to Ca flux when Ca is the sensor.

# Excitatory
# no H
exp280:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp280.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp280_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=False' \
	::: 3 4 ::: 100 101 105 110 115 120 125 130 | tee $(DATA_PATH)/exp280.log

# H
exp281:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp281.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp281_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=True' \
	::: 3 4 ::: 100 101 105 110 115 120 125 130 | tee $(DATA_PATH)/exp281.log


# ----------------------------------------------------------------------------
# Based on 280-1 results (which require working on an old branch)
# eea7799b9f1d20b9fe9d6f412b1258d37d69a501

# Here:
# - I exposed G_KCa, and am going zero it out here. 
#
# This should return expected H+AMPA effect. 
# - Note: I arb. choose the value of G_KCa that disrupted the expected effects. 
# - Need to do sensitivity tests on this important param, once I confirm 
# effect recovery with these two runs.

# SUM: Huh. H effect abolished again. It's not G_KCa afterall? Or, bug?

# Excitatory
# no H
exp282:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp282.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp282_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --G_KCa=0 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=False' \
	::: 3 4 ::: 100 101 105 110 115 120 125 130 | tee $(DATA_PATH)/exp282.log

# H
exp283:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp283.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp283_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --G_KCa=0 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=True' \
	::: 3 4 ::: 100 101 105 110 115 120 125 130 | tee $(DATA_PATH)/exp283.log


# ----------------------------------------------------------------------------
# Debug conts.
# 5d85c2da11476471f1f2ac23a4176ea8b36e5f5d

# Commented out d_Ca dynamics.

# SUM: TADA! Obv. Ca flux cancels out all other channel effects. Need to do sensitivity for this as part of the control figures.

# Excitatory
# no H
exp284:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp284.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp284_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --G_KCa=0 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=False' \
	::: 3 4 ::: 100 101 105 110 115 120 125 130 | tee $(DATA_PATH)/exp284.log

# H
exp285:
	parallel -j 12 -v \
		--joblog '$(DATA_PATH)/exp285.parallel.log' \
		--nice 19 --delay 2 --colsep ',' \
	'python homeostasis_experiments.py run $(DATA_PATH)/exp285_stim{1}_osc{2} --stim_name=$(DATA_PATH)/stim{1}.csv --osc_name=$(DATA_PATH)/osc{2}.csv --N=100 --t=20 --tau_h=4 --burn_t=19.5 --Ca_target=0.003 --G_KCa=0 --w_min=5e-6 --w_max=50e-6 --sigma=0 --num_trials=10 --seed_value=42 --verbose --homeostasis=True' \
	::: 3 4 ::: 100 101 105 110 115 120 125 130 | tee $(DATA_PATH)/exp285.log

