# Make spike trains for later exps
make lfp100 oscsix100 oscseven100 oscnine100 oscten100 lowgamma100 highgamma100 ripple100

# Sample CA1 LFP (shape control)
make exp244 exp245 

# Explore gamma=60Hz, gamma=90Hz, ripple=150 Hz AND
make exp249 exp248 exp250 exp251 exp252 exp253

# Explore freq sensitivity near theta
make exp254 exp255 exp258 exp259 exp262 exp263 exp266 exp267 

# Explore param sensitivities fo g_Ca, g_L, noise
make exp270 exp271 exp272 exp273 exp274 exp275 

# Joint:
make exp244 exp245 exp248 exp250 exp252 exp254 exp255 exp258 exp259 exp262 exp263 exp266 exp267 exp270 exp271 exp272 exp273 exp274 exp275 

# Missed controls (DONE, TRANSFERRED)
make exp249 exp251 exp253