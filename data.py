{
    'kp_loc_norm': tensor([[[-0.0902,  0.1394,  0.0675,  0.0577, -0.3237, -0.2352, -0.0626,
                           -0.0812, -0.0837, -0.0924, -0.0336, -0.4095, -0.5862, -0.5544,
                           0.2748,  0.8037,  1.2100],
                            [0.1404,  0.1093,  1.0279,  1.9415,  0.1723,  1.0969,  2.0024,
                             -0.3649, -0.8704, -1.0781, -1.2775, -0.7702, -0.2002,  0.2714,
                             -0.8340, -0.6574, -0.7092]]], device='cuda:0'),
    'kp_mean': tensor([[0.0902, -0.1404]], device='cuda:0'),
    'phi': {'R_log': tensor([[-0.1018,  2.7115,  0.5542]], device='cuda:0', grad_fn= < SelectBackward0 > ),
            'R': tensor([[[-0.9288, -0.1423,  0.3418],
                          [0.0033,  0.9201,  0.3918],
                          [-0.3702,  0.3650, -0.8541]]], device='cuda:0', grad_fn= < AddBackward0 >),
            'scale': tensor([1.], device='cuda:0'), 'T': tensor([[0., 0., 0.]], device='cuda:0'),
            'shape_camera_coord': tensor([[[-0.0715,  0.1948,  0.2110,  0.0829, -0.3271, -0.2251, -0.2782,
                                            -0.0829, -0.0868,  0.0526, -0.0086, -0.3636, -0.6287, -0.6015,
                                            0.1929,  0.7584,  1.2125],
                                           [0.1456,  0.1223,  1.0766,  1.9538,  0.1705,  1.1183,  1.9811,
                                            -0.3674, -0.9089, -1.0474, -1.2789, -0.7456, -0.2271,  0.2572,
                                            -0.8446, -0.6374, -0.7320],
                                           [0.0840,  0.2945,  0.3979,  0.7644, -0.1155, -0.2319, -0.0493,
                                            0.0630, -0.0876, -0.2189, -0.1735, -0.2370, -0.4426, -0.8497,
                                            0.1591,  0.4209,  0.2732]]], device='cuda:0', grad_fn= < AddBackward0 >),
            'shape_coeff': tensor([[3.5105, -3.0565,  0.1834,  1.5520, -3.1669,  2.9971, -1.7748, -2.0579,
                                    0.4650, -1.2811]], device='cuda:0', grad_fn= < SelectBackward0 > ),
            'shape_canonical': tensor([[[0.0358, -0.2896, -0.3399, -0.3537,  0.3472,  0.2987,  0.2832,
                                         0.0525,  0.1101,  0.0288,  0.0680,  0.4231,  0.7472,  0.8743,
                                         -0.2409, -0.8625, -1.2300],
                                        [0.1748,  0.1923,  1.1058,  2.0648,  0.1612,  0.9762,  1.8442,
                                         -0.3032, -0.8558, -1.0510, -1.2387, -0.7207, -0.2811,  0.0119,
                                         -0.7464, -0.5406, -0.7463],
                                        [-0.0391, -0.1371,  0.1540,  0.1408,  0.0537,  0.5593,  0.7233,
                                         -0.2262, -0.3110, -0.2054, -0.3559, -0.2139,  0.0742,  0.6210,
                                         -0.4009, -0.3501, -0.1057]]], device='cuda:0', grad_fn= < ViewBackward0 >)},
    'l_canonicalization': tensor(0.0100, device='cuda:0', grad_fn= < MeanBackward0 >),
    'psi': {'shape_coeff': tensor([[[3.4484, -3.1329,  0.0321,  1.4859, -3.0985,  3.0536, -1.7140,
                                  -2.1183,  0.5237, -1.2724]],

                                   [[3.5261, -3.0303,  0.0996,  1.6008, -3.1813,  2.9604, -1.7530,
                                     -1.9989,  0.6293, -1.2853]],

                                   [[3.5305, -3.1904,  0.1541,  1.4146, -3.1899,  2.9678, -1.7455,
                                     -1.9723,  0.5670, -1.4342]],

                                   [[3.5075, -3.0740,  0.1130,  1.5508, -3.1287,  2.9981, -1.7052,
                                     -2.1196,  0.4838, -1.2974]]], device='cuda:0', grad_fn= < ViewBackward0 >),
            'shape_canonical': tensor([[[[0.0290, -0.2942, -0.3392, -0.3391,  0.3380,  0.2855,  0.2763,
                                      0.0528,  0.1105,  0.0268,  0.0678,  0.4221,  0.7510,  0.8926,
                                      -0.2366, -0.8502, -1.2402],
                                         [0.1717,  0.1876,  1.0989,  2.0558,  0.1598,  0.9732,  1.8385,
                                      -0.3031, -0.8567, -1.0563, -1.2415, -0.7191, -0.2786,  0.0196,
                                      -0.7502, -0.5364, -0.7153],
                                         [-0.0210, -0.1193,  0.1786,  0.1408,  0.0711,  0.5713,  0.7077,
                                      -0.2259, -0.3138, -0.1942, -0.3501, -0.2198,  0.0463,  0.6121,
                                      -0.4064, -0.3735, -0.1173]]],


                                       [[[0.0359, -0.2894, -0.3488, -0.3649,  0.3473,  0.3003,  0.2893,
                                          0.0566,  0.1101,  0.0279,  0.0672,  0.4235,  0.7432,  0.8456,
                                          -0.2393, -0.8498, -1.2011],
                                         [0.1731,  0.1885,  1.1023,  2.0696,  0.1617,  0.9798,  1.8590,
                                           -0.3090, -0.8668, -1.0633, -1.2535, -0.7289, -0.2837,  0.0168,
                                           -0.7554, -0.5364, -0.7070],
                                           [-0.0484, -0.1480,  0.1671,  0.1608,  0.0460,  0.5598,  0.7177,
                                            -0.2369, -0.3146, -0.1969, -0.3509, -0.2168,  0.0703,  0.6285,
                                            -0.4079, -0.3598, -0.0848]]],


                                       [[[0.0366, -0.2922, -0.3434, -0.3532,  0.3514,  0.2875,  0.2634,
                                          0.0538,  0.1089,  0.0216,  0.0614,  0.4261,  0.7567,  0.8651,
                                          -0.2431, -0.8512, -1.1965],
                                         [0.1785,  0.1964,  1.1115,  2.0921,  0.1650,  0.9829,  1.8707,
                                           -0.3061, -0.8700, -1.0651, -1.2570, -0.7306, -0.2710, -0.0059,
                                           -0.7579, -0.5258, -0.7608],
                                           [-0.0413, -0.1383,  0.1790,  0.1659,  0.0506,  0.5832,  0.7488,
                                            -0.2317, -0.3057, -0.1868, -0.3376, -0.2147,  0.0451,  0.5821,
                                            -0.3993, -0.3770, -0.1387]]],


                                       [[[0.0302, -0.2933, -0.3423, -0.3500,  0.3400,  0.2940,  0.2869,
                                          0.0514,  0.1098,  0.0281,  0.0678,  0.4217,  0.7501,  0.8819,
                                          -0.2387, -0.8570, -1.2263],
                                         [0.1739,  0.1904,  1.1053,  2.0596,  0.1613,  0.9806,  1.8470,
                                           -0.3023, -0.8548, -1.0523, -1.2384, -0.7194, -0.2853,  0.0112,
                                           -0.7472, -0.5436, -0.7392],
                                           [-0.0276, -0.1264,  0.1680,  0.1472,  0.0656,  0.5579,  0.7045,
                                            -0.2244, -0.3137, -0.2036, -0.3564, -0.2161,  0.0668,  0.6228,
                                            -0.4056, -0.3635, -0.1113]]]], device='cuda:0', grad_fn= < ViewBackward0 > )},
    'kp_reprojected': tensor([[[-0.0715,  0.1948,  0.2110,  0.0829, -0.3271, -0.2251, -0.2782,
                                -0.0829, -0.0868,  0.0526, -0.0086, -0.3636, -0.6287, -0.6015,
                                0.1929,  0.7584,  1.2125],
                               [0.1456,  0.1223,  1.0766,  1.9538,  0.1705,  1.1183,  1.9811,
                                -0.3674, -0.9089, -1.0474, -1.2789, -0.7456, -0.2271,  0.2572,
                                -0.8446, -0.6374, -0.7320]]], device='cuda:0', grad_fn= < SliceBackward0 > ),
    'l_reprojection': tensor(0.0520, device='cuda:0', grad_fn= < MeanBackward0 >),
    'kp_reprojected_image': tensor([[[0.0000,  0.2296,  0.1577,  0.1479, -0.2335, -0.1450,  0.0276,
                                   0.0090,  0.0065, -0.0022,  0.0566, -0.3193, -0.4960, -0.4642,
                                   0.3650,  0.8939,  1.3002],
                                     [0.0000, -0.0311,  0.8875,  1.8011,  0.0319,  0.9565,  1.8620,
                                   -0.5053, -1.0108, -1.2185, -1.4179, -0.9106, -0.3406,  0.1310,
                                   -0.9744, -0.7978, -0.8496]]], device='cuda:0', grad_fn= < AddBackward0 >),
    'shape_image_coord': tensor([[[0.0000,  0.2296,  0.1577,  0.1479, -0.2335, -0.1450,  0.0276,
                                0.0090,  0.0065, -0.0022,  0.0566, -0.3193, -0.4960, -0.4642,
                                0.3650,  0.8939,  1.3002],
                                  [0.0000, -0.0311,  0.8875,  1.8011,  0.0319,  0.9565,  1.8620,
                                -0.5053, -1.0108, -1.2185, -1.4179, -0.9106, -0.3406,  0.1310,
                                -0.9744, -0.7978, -0.8496],
                                  [0.0840,  0.2945,  0.3979,  0.7644, -0.1155, -0.2319, -0.0493,
                                0.0630, -0.0876, -0.2189, -0.1735, -0.2370, -0.4426, -0.8497,
                                0.1591,  0.4209,  0.2732]]], device='cuda:0', grad_fn= < CatBackward0 > ),
    'objective': tensor(0.0620, device='cuda:0', grad_fn= < SumBackward0 >)
}