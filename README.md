# resistingrhythm

A study of modulation, oscillation, and homeostasis.

# Publications

This work was presented at SFN2018. A copy of the poster can be found [here](). To re-run all experiments:

1. Install this package and its dependencies (see below).
2. At the command line, and from `\exp` directory, run 

    - `make stim3 stim4 osc100 burst100` then
    - `make exp210 exp211 exp212 exp213 exp214 exp215 exp216 exp217 exp218`. 

     - _Note_: you may need to adjust the `$DATA_PATH` variable in the [Makefile](https://github.com/voytekresearch/resistingrhythm/blob/master/resistingrhythm/exp/Makefile).

    - The experimental recipes rely on gnu parallel, and are configured to a 40 core machine. If you have more or fewer cores, adjust the `-j 38` argument in each recipe accordingly. The Makefile is [here](https://github.com/voytekresearch/resistingrhythm/blob/master/resistingrhythm/exp/Makefile). 
    - _Note_: with the current configuration these simulations take about 4 days.
   

3. Once (2) is complete open `papers_figures_v3.Rmd` (found [here](https://github.com/voytekresearch/resistingrhythm/blob/master/resistingrhythm/analysis/paper_figures_v3.Rmd)), adjust the `data_path`, and execute all cells.
4. To generate the example traces, open `testing_HHH.ipynb` (found [here](https://github.com/voytekresearch/resistingrhythm/blob/master/resistingrhythm/ipynb/testing_HHH.ipynb)) and re-run all cells. This should take half an hour or so. Step (4) can be completed anytime; it is not dependent on 1-3.

# Installation


From the command line (on linux or macOS) run,
- `git clone git@github.com:voytekresearch/resistingrhythm.git` 

then

-  `cd resistingrhythm; pip install -e .`.

# Dependencies
- A standard Python >3.6 anaconda install (https://www.anaconda.com/download)
- brian2 (https://brian2.readthedocs.io/en/stable/)
- fire (https://github.com/google/python-fire)
- fakespikes (https://github.com/voytekresearch/fakespikes)
- make (the standard unix utility)
- GNU parallel (https://www.gnu.org/software/parallel/)
