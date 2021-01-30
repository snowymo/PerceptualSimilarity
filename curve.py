import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams["font.family"] = "Arial"


a = np.sort(np.random.rand(6,18), axis=0)
x = np.arange(len(a[0]))

all_data=[]

input_f = open("lpips_curve_125_bedroom.csv", "r")
for line in input_f:
    if not line.startswith("scene"):
        s = line.strip().split(',')
        all_data.append(np.array(s[0:6]))

all_data = np.array(all_data)

handles = []
labels=[]

step = 5
NUM_ECC = int((105-5)/step+1)
scenes=["bedroom", "gas", "lobby", "mc", "gallery"]
scenes=["bedroom"]
for scene in range(len(scenes)): # scene
    if scene == 2:
        continue
    plt.close("all")
    fig, ax = plt.subplots(figsize=(8,4))

    data_our = []
    data_nerf = []
    data_fovea = []

    for i in range(NUM_ECC): # of eccentricities
        data_our.append([])
        data_nerf.append([])
        data_fovea.append([])

    for d in all_data: #data entry
        if int(float(d[0])) == int(float(scene)):
            ecc_ind = int(int(float(d[1]))/step-1)
            data_our[ecc_ind].append(float(d[2]))
            data_nerf[ecc_ind].append(float(d[3]))
            data_fovea[ecc_ind].append(float(d[4]))

    data_our = np.array(data_our)
    data_nerf = np.array(data_nerf)
    data_fovea = np.array(data_fovea)

    data_our_plot = []
    data_nerf_plot = []
    data_fovea_plot = []

    for e in range(np.shape(data_our)[0]): #inside each ecc
        data_our[e] = np.array(data_our[e])
        data_nerf[e] = np.array(data_nerf[e])
        data_fovea[e] = np.array(data_fovea[e])

        data_our_plot.append([data_our[e].mean(), data_our[e].std(), np.percentile(data_our[e], 75), np.percentile(data_our[e], 25), np.max(data_our[e]), np.min(data_our[e])])
        data_nerf_plot.append([data_nerf[e].mean(), data_nerf[e].std(), np.percentile(data_nerf[e], 75), np.percentile(data_nerf[e], 25), np.max(data_nerf[e]), np.min(data_nerf[e])])
        data_fovea_plot.append([data_fovea[e].mean(), data_fovea[e].std(), np.percentile(data_fovea[e], 75), np.percentile(data_fovea[e], 25), np.max(data_fovea[e]), np.min(data_fovea[e])])

    data_our_plot = np.array(data_our_plot)
    data_nerf_plot = np.array(data_nerf_plot)
    data_fovea_plot = np.array(data_fovea_plot)

    #for i in range(1): #OUR, NERF
    X = np.arange(NUM_ECC) * step + step

    ax.plot(X, data_our_plot[:,0], color='blue', lw=3, label="OURS", zorder=4)
    ax.plot(X, data_nerf_plot[:,0], color='orange', lw=3, label="NeRF", zorder=3)
    ax.plot(X, data_fovea_plot[:,0], color='green', lw=3, label="F-GT", zorder=3)

    # intersection point
    idx = np.argwhere(np.diff(np.sign(data_our_plot[:,0] - data_nerf_plot[:,0]))).flatten()
    ax.plot(X[idx], data_our_plot[:,0][idx], 'rx', zorder=5)
    idx = np.argwhere(np.diff(np.sign(data_our_plot[:, 0] - data_fovea_plot[:, 0]))).flatten()
    ax.plot(X[idx], data_our_plot[:, 0][idx], 'r+', zorder=5)

    ax.fill_between(X, data_our_plot[:,2], data_our_plot[:,3],color='blue', alpha=0.4, lw=0, zorder=2)
    #ax.fill_between(X, data_our_plot[:,4], data_our_plot[:,5],color='orange', alpha=0.15, lw=0, zorder=2)
    #ax.fill_between(X, data_our_plot[:,0]+data_our_plot[:,1], data_our_plot[:,0]-data_our_plot[:,1],color='orange', alpha=0.15, lw=0, zorder=1)
    
    ax.fill_between(X, data_nerf_plot[:,2], data_nerf_plot[:,3],color='orange', alpha=0.15, lw=0, zorder=2)
    #ax.fill_between(X, data_nerf_plot[:,4], data_nerf_plot[:,5],color='green', alpha=0.15, lw=0, zorder=1)

    ax.fill_between(X, data_fovea_plot[:,2], data_fovea_plot[:,3],color='green', alpha=0.15, lw=0, zorder=2)
    #ax.fill_between(X, data_fovea_plot[:,4], data_fovea_plot[:,5],color='blue', alpha=0.15, lw=0, zorder=1)

    ax.set_xlim([5,105])
    major_ticks = [5, 20, 40, 60, 80, 100]
    # major_ticks = np.arange(5,105,2)
    ax.set_xticks(major_ticks)
    ax.set_ylim([0,0.75])
    ax.xaxis.grid(which='major', linestyle=':', zorder=10)  # vertical lines
    ax.yaxis.grid(which='major', linestyle=':', zorder=10)
    for k, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_zorder(10)
    ax.tick_params(which='both', direction="in", zorder=10)
    for tic in ax.yaxis.get_major_ticks():
        tic.label.set_fontsize(14)
        tic.tick1line.set_visible = tic.tick2line.set_visible = False
    for tic in ax.xaxis.get_major_ticks():
        tic.label.set_fontsize(14)
        tic.tick1line.set_visible = True
        tic.tick2line.set_visible = False

    ax.set_xlabel("Eccentricity (deg)",fontsize=14)
    ax.set_ylabel("LPIPS",fontsize=14)
    # if scene == 1:
    # plt.legend(fancybox=True, fontsize=14)
    # plt.show()
    plt.savefig("lpips_scene_test_"+scenes[int(float(scene))]+".pdf",bbox_inches='tight')
