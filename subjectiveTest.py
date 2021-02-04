# analyze binom_test to each pair
# for further analyze ANOVA
# gt/-our	gt/-nerf	-our/gt	-our/nerf	-nerf/gt	nerf/-our
# 40	    1	    38	    77	        1	    75
# 78	    78	    78	    78	        78	    78

from scipy import stats
import numpy as np

# choose us, or nerf is no us
choose = [64, 110, 111]
stimuli = ["our-gt", "our-nerf", "nerf-gt"]
total = 60*2
binresult = []

for eachChoose in choose:
    prob = stats.binom_test(eachChoose, n=total, p=0.5, alternative="greater")
    print(prob)
    binresult.append(prob)

np.savetxt("bintest.csv", binresult)