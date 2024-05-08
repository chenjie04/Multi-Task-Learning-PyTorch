import matplotlib.pyplot as plt

import numpy as np 

x = np.arange(0, 30, 1)
print(x)

train_seg_groupnet = np.array([16.3406, 28.4913,36.3585,42.8879,48.5363,52.8737,
                      57.4788,61.9201,65.1205,68.1912,72.0059,73.7001,76.0988,
                      77.6330, 79.7797,80.5245,82.8616,83.8319,84.7174,85.8949,
                      86.4330,87.4944,88.2769,88.4983,89.5756,90.1693,90.3685,
                      90.9820,91.4797,91.6849])
val_seg_groupnet = np.array([23.5902, 30.8178,36.2199,40.5642,42.4792,43.4562,
                    45.5456,47.1662,46.0289,47.4732,47.9451,49.7601,48.7846,
                    48.6492,50.1891,50.8678,49.5582,49.8297,51.3416,52.4344,
                    52.2923,52.5189,52.9869,53.0436,53.6271,53.3809,53.8068,
                    53.8063,54.3750,54.4155])

train_human_parts_groupnet = np.array([38.3948,47.8394,52.9477,57.6533,61.4190,
                              63.9917, 67.3377,69.3294,71.5970,73.1713,74.9047,
                              76.1591,77.8976,78.7598,79.3395,80.7551, 81.2724,
                              82.4150,83.0944,83.4358,84.3877,85.0241,85.4176,
                              86.1041,86.5811,86.9850,87.2787,87.8197,88.1483,
                              88.3668])
val_human_parts_groupnet = np.array([38.0695,45.8928,49.2259,51.2151,52.4488,
                            53.6245,55.3889,55.4765,53.1335, 56.3195,55.0506,
                            56.7900,56.7682,56.5781,56.8012,56.8380,57.1165,
                            57.0933,57.1781,57.8568,57.7708,57.9168,57.4950,
                            57.7737,58.1702,57.9044,58.2115,58.0821,58.3504,58.3751])

train_saliency_groupnet = np.array([54.611,59.876,61.985,64.048,65.680,66.882,68.362,
                           69.438,70.656,71.757,72.923,73.729,74.715,75.504,
                           76.119,76.917,77.709,78.400,79.041,79.727,80.249,
                           80.896,81.532,82.060,82.510,82.942,83.454,83.915,
                           84.199,84.475])
val_saliency_groupnet = np.array([56.067,58.470,59.175,59.885,61.981,62.094,60.790,
                         61.176,61.624,63.057,62.362,62.498,62.296,62.377,63.494,
                         63.378,63.624,63.562,63.009,64.231,63.557,63.284,64.449,
                         63.978,63.859,64.119,63.803,63.795,64.164,63.913])

train_normal_groupnet = np.array([22.3438,20.7238,19.8023,19.1377,18.7292,18.4296,17.9853,
                          17.6110,17.3648,17.0720,16.8395,16.5791,16.2902,
                          16.1311,15.9010,15.6026,15.4047,15.1900,14.9665,14.8007,
                          14.5454,14.3898,14.2070,14.0019,13.7576,13.6543,13.5391,
                          13.3290,13.1843,13.0878])
val_normal_groupnet = np.array([18.0666,16.5708,17.1676,15.6524,15.3696,15.7765,15.4540,
                       15.1640,15.2258,14.5234,14.6557,14.4876,14.8736,14.6196,
                       14.4074,14.5463,14.5510,14.4533,14.3474,14.6106,14.4232,
                       14.1737,14.4673,14.2345,14.0065,14.1384,14.1063,14.0747,
                       13.9474,13.9427])

# ---------------------------------------------------------------------------------

train_seg_nddr = np.array([64.4401,80.9127,85.3599,88.3884,89.0788,90.1911,90.5381,91.2657,
                           91.7851,92.0198,92.4502,92.4691,93.0112,93.2315,93.3837,93.3308,
                           93.7453,93.9556,94.1006,94.1512,94.1258,94.4784,94.4906,94.6400,
                           94.6767,94.7584,94.8363,94.9230,94.9686,94.9840])
val_seg_nddr = np.array([40.4560,46.7428,46.9607,49.5704,49.6134,49.7568,50.3852,50.6753,
                         50.9258,50.2374,50.3909,50.7189,50.5455,51.3584,50.9534,51.0850,
                         51.0345,51.9084,51.2530,51.3664,51.4196,51.5827,51.6376,51.9651,
                         51.9641,51.8565,52.0040,51.7831,51.7902,51.8814])

train_human_parts_nddr = np.array([49.0794,62.9557,72.4734,78.0710,80.8071,82.5958,82.7541,
                                   84.9671,85.5163,86.0415,86.8489,87.0982,87.6368,88.0249,
                                   88.2947,88.5757,88.8432,89.2887,89.4942,89.7421,89.7509,
                                   90.1716,90.3797,90.5025,90.6615,90.7905,90.9317,91.0274,
                                   91.1601,91.1917])
val_human_parts_nddr = np.array([43.3492,50.5293,51.5719,53.6865,53.5539,54.6715,54.5889,
                                 54.6794,54.2192,54.3554,54.8260,54.8975,55.0362,54.8919,
                                 54.8845,54.8881,54.9471,54.7095,54.9335,54.8776,54.9414,
                                 55.0322,54.9680,54.9109,54.8862,54.9555,55.0825,54.9604,
                                 55.0171,55.1285])

train_saliency_nddr = np.array([64.213,69.512,73.187,75.669,77.640,79.259,80.059, 81.198,
                                82.400,82.819,83.784,83.971,84.620,85.268,85.582,85.875,
                                86.315,86.627,86.950,87.174,87.459,87.762,87.838,87.969,
                                88.237,88.375,88.498,88.614,88.762,88.870])
val_saliency_nddr = np.array([61.415,61.061,62.377,62.978,62.852,63.366,63.390,62.960,
                              63.559,63.812,63.202,63.981,64.094,63.738,63.839,64.005,
                              63.957,64.226,64.298,63.906,64.151,64.073,64.212,64.280,
                              64.246,64.294,64.256,64.267,64.209,64.221])

train_normal_nddr = np.array([19.1116,17.5769,16.7485,16.1003,15.5687,15.0346, 14.7753,
                              14.3698,14.0491,13.8063,13.4690,13.3603,13.1035,12.8931,
                              12.6949,12.5401,12.3845,12.1956,12.0198,11.9144,11.8242,
                              11.6291,11.4999,11.4193,11.2868,11.2485,11.1164,11.0256,
                              10.9738,10.9394])
val_normal_nddr = np.array([16.8769,15.0658,14.9204,14.9587,14.9112,14.6260,14.5431,
                            14.1009,14.4604,14.2243,14.1800,14.3997,14.1256,14.3459,
                            14.1021,14.1704,14.2097,14.2086,14.1912,14.1724,14.1704,
                            14.1689,14.1673,14.2271,14.1432,14.2151,14.2173,14.2285,
                            14.2619,14.1768])

plt.subplot(2, 2, 1)
plt.plot(x, train_seg_groupnet, label='train_SquadNet', color='#14517c', marker='o', markersize=3,markevery=3)
plt.plot(x, val_seg_groupnet, label='val_SquadNetL',color='#14517c', marker='*',markersize=3,markevery=3)
plt.plot(x, train_seg_nddr, label='train_NDDR-CNN', color='#f3d266', marker='o', markersize=3,markevery=3)
plt.plot(x, val_seg_nddr, label='val_NDDR-CNN', color='#f3d266', marker='*', markersize=3,markevery=3)
plt.legend(loc="lower left", prop={'size': 6})
plt.ylabel('mIoU')
plt.title('Segmentation', y=-0.3, fontsize=10)

plt.subplot(2, 2, 2)
plt.plot(x, train_human_parts_groupnet, label='train_SquadNet',color='#14517c', marker='o', markersize=3,markevery=3)
plt.plot(x, val_human_parts_groupnet, label='val_SquadNet',color='#14517c', marker='*',markersize=3,markevery=3)
plt.plot(x, train_human_parts_nddr, label='train_NDDR-CNN', color='#f3d266', marker='o', markersize=3,markevery=3)
plt.plot(x, val_human_parts_nddr, label='val_NDDR-CNN', color='#f3d266', marker='*', markersize=3,markevery=3)
plt.legend(loc="lower left", prop={'size': 6})
plt.ylabel('mIoU')
plt.title('Human Parts', y=-0.3,fontsize=10)

plt.subplot(2, 2, 3)
plt.plot(x, train_saliency_groupnet, label='train_SquadNet',color='#14517c', marker='o', markersize=3,markevery=3)
plt.plot(x, val_saliency_groupnet, label='val_SquadNet',color='#14517c', marker='*',markersize=3,markevery=3)
plt.plot(x, train_saliency_nddr, label='train_NDDR-CNN', color='#f3d266', marker='o', markersize=3,markevery=3)
plt.plot(x, val_saliency_nddr, label='val_NDDR-CNN', color='#f3d266', marker='*', markersize=3,markevery=3)
plt.legend(loc="lower left", prop={'size': 6})
plt.ylabel('mIoU')
plt.title('Saliency', y=-0.3,fontsize=10)

plt.subplot(2, 2, 4)
plt.plot(x, train_normal_groupnet, label='train_SquadNet',color='#14517c', marker='o', markersize=3,markevery=3)
plt.plot(x, val_normal_groupnet, label='val_SquadNet',color='#14517c', marker='*',markersize=3,markevery=3)
plt.plot(x, train_normal_nddr, label='train_NDDR-CNN', color='#f3d266', marker='o', markersize=3,markevery=3)
plt.plot(x, val_normal_nddr, label='val_NDDR-CNN', color='#f3d266', marker='*', markersize=3,markevery=3)
plt.legend(loc="lower left", prop={'size': 6})
plt.ylabel('mErr')
plt.title('Normal', y=-0.3,fontsize=10)

plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)

plt.savefig('training_curve_nddr_groupnet.pdf')