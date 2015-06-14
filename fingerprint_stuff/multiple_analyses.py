__author__ = 'jeremy'
import numpy as np

import rate_fp
import fingerprint_core as fp_core
import NNSearch
import constants

fingerprint_length = constants.fingerprint_length
max_items = 50
histogram_weights = 2.0 * np.ones(constants.histograms_length * 2)
entropy_weights = np.ones(3)
energy_weights = np.ones(3)
weights = np.concatenate((energy_weights, entropy_weights, histogram_weights))
rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp, weights=weights,
                            distance_function=NNSearch.distance_1_k, distance_power=0.5, n_docs=max_items,
                            filename='011.gcfp_histweighted')

histogram_weights = 3.0 * np.ones(constants.histograms_length * 2)
entropy_weights = np.ones(3)
energy_weights = np.ones(3)
weights = np.concatenate((energy_weights, entropy_weights, histogram_weights))
rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp, weights=weights,
                            distance_function=NNSearch.distance_1_k, distance_power=0.5, n_docs=max_items,
                            filename='012.gcfp_histweighted')

histogram_weights = 5.0 * np.ones(constants.histograms_length * 2)
entropy_weights = np.ones(3)
energy_weights = np.ones(3)
weights = np.concatenate((energy_weights, entropy_weights, histogram_weights))
rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp, weights=weights,
                            distance_function=NNSearch.distance_1_k, distance_power=0.5, n_docs=max_items,
                            filename='013.gcfp_histweighted')

histogram_weights = np.ones(constants.histograms_length * 2)
entropy_weights = 2.0 * np.ones(3)
energy_weights = 2.0 * np.ones(3)
weights = np.concatenate((energy_weights, entropy_weights, histogram_weights))
rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp, weights=weights,
                            distance_function=NNSearch.distance_1_k, distance_power=0.5, n_docs=max_items,
                            filename='014.gcfp.histdeweighted')

histogram_weights = np.ones(constants.histograms_length * 2)
entropy_weights = 3.0 * np.ones(3)
energy_weights = 3.0 * np.ones(3)
weights = np.concatenate((energy_weights, entropy_weights, histogram_weights))
rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp, weights=weights,
                            distance_function=NNSearch.distance_1_k, distance_power=0.5, n_docs=max_items,
                            filename='016.gcfp.histdeweighted')

histogram_weights = np.ones(constants.histograms_length * 2)
entropy_weights = 5.0 * np.ones(3)
energy_weights = 5.0 * np.ones(3)
weights = np.concatenate((energy_weights, entropy_weights, histogram_weights))
rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp, weights=weights,
                            distance_function=NNSearch.distance_1_k, distance_power=0.5, n_docs=max_items,
                            filename='017.gcfp.histdeweighted')

#########
# max_items = 50
# DONE
#########
if (0):


    histogram_weights = np.ones(constants.histograms_length * 2)
    entropy_weights = 1.5 * np.ones(3)
    energy_weights = 1.5 * np.ones(3)
    weights = np.concatenate((energy_weights, entropy_weights, histogram_weights))

    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.regular_fp, weights=weights,
                                distance_function=NNSearch.distance_1_k, distance_power=0.5, n_docs=max_items,
                                filename='gcfp.weights1.5')

    histogram_weights = np.ones(constants.histograms_length * 2)
    entropy_weights = 2 * np.ones(3)
    energy_weights = 2 * np.ones(3)
    weights = np.concatenate((energy_weights, entropy_weights, histogram_weights))

    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp, weights=weights,
                                distance_function=NNSearch.distance_1_k, distance_power=0.5, n_docs=max_items,
                                filename='gcfp.weights2')

    histogram_weights = np.ones(constants.histograms_length * 2)
    entropy_weights = 3 * np.ones(3)
    energy_weights = 3 * np.ones(3)
    weights = np.concatenate((energy_weights, entropy_weights, histogram_weights))

    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp, weights=weights,
                                distance_function=NNSearch.distance_1_k, distance_power=0.5, n_docs=max_items,
                                filename='gcfp.weights3')
    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp_with_kwargs, weights=np.ones(fingerprint_length),
                            distance_function=NNSearch.distance_1_k, distance_power=0.5, n_docs=max_items,
                            filename='gcfp.h=120', fingerprint_length=120)

    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp_with_kwargs, weights=np.ones(fingerprint_length),
                            distance_function=NNSearch.distance_1_k, distance_power=0.5, n_docs=max_items,
                            filename='gcfp.h=120', fingerprint_length=120)

    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp_with_kwargs, weights=np.ones(fingerprint_length),
                                distance_function=NNSearch.distance_1_k, distance_power=0.5, n_docs=max_items,
                                filename='gcfp.h=40', fingerprint_length=40)

    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp_with_kwargs, weights=np.ones(fingerprint_length),
                                distance_function=NNSearch.distance_1_k, distance_power=0.5, n_docs=max_items,
                                filename='gcfp.h=80', fingerprint_length=80)

    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp_with_kwargs, weights=np.ones(fingerprint_length),
                                distance_function=NNSearch.distance_1_k, distance_power=0.5, n_docs=max_items,
                                filename='gcfp.h=10', fingerprint_length=10)

    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp_with_kwargs, weights=np.ones(fingerprint_length),
                                distance_function=NNSearch.distance_1_k, distance_power=0.5, n_docs=max_items,
                                filename='gcfp.h=20', fingerprint_length=20)


    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp_bw, weights=np.ones(fingerprint_length),
                            distance_function=NNSearch.distance_1_k,
                            distance_power=0.5, n_docs=max_items, use_visual_output1=False,
                            use_visual_output2=False, image_sets=None, self_reporting=None, comparisons_to_make=None,
                            filename='gcfpbw.k0.5')

    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp, weights=np.ones(fingerprint_length),
                                distance_function=NNSearch.distance_1_k,
                                distance_power=0.5, n_docs=max_items, use_visual_output1=False,
                                use_visual_output2=False, image_sets=None, self_reporting=None,
                                comparisons_to_make=None,
                                filename='gcfp.k0.5')

    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.regular_fp, weights=np.ones(fingerprint_length),
                                distance_function=NNSearch.distance_1_k,
                                distance_power=0.5, n_docs=max_items, use_visual_output1=False,
                                use_visual_output2=False, image_sets=None, self_reporting=None,
                                comparisons_to_make=None,
                                filename='fp_k0.5')

    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp, weights=np.ones(fingerprint_length),
                                distance_function=NNSearch.distance_1_k,
                                distance_power=0.7, n_docs=max_items, use_visual_output1=False,
                                use_visual_output2=False, image_sets=None, self_reporting=None,
                                comparisons_to_make=None,
                                filename='gcfp.k0.7')

    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp, weights=np.ones(fingerprint_length),
                                distance_function=NNSearch.distance_1_k,
                                distance_power=0.9, n_docs=max_items, use_visual_output1=False,
                                use_visual_output2=False, image_sets=None, self_reporting=None,
                                comparisons_to_make=None,
                                filename='gcfp.k0.9')

    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp, weights=np.ones(fingerprint_length),
                                distance_function=NNSearch.distance_1_k,
                                distance_power=1.1, n_docs=max_items, use_visual_output1=False,
                                use_visual_output2=False, image_sets=None, self_reporting=None,
                                comparisons_to_make=None,
                                filename='gcfp.k1.1')

    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp, weights=np.ones(fingerprint_length),
                                distance_function=NNSearch.distance_1_k,
                                distance_power=0.3, n_docs=max_items, use_visual_output1=False,
                                use_visual_output2=False, image_sets=None, self_reporting=None,
                                comparisons_to_make=None,
                                filename='gcfp.k1.3')

    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp_bw, weights=np.ones(fingerprint_length),
                                distance_function=NNSearch.distance_1_k,
                                distance_power=0.5, n_docs=max_items, use_visual_output1=False,
                                use_visual_output2=False, image_sets=None, self_reporting=None,
                                comparisons_to_make=None,
                                filename='gcfpbw.k0.5')

    max_items = 70
    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp_bw, weights=np.ones(fingerprint_length),
                                distance_function=NNSearch.distance_1_k,
                                distance_power=0.5, n_docs=max_items, use_visual_output1=False,
                                use_visual_output2=False, image_sets=None, self_reporting=None,
                                comparisons_to_make=None,
                                filename='gcfpbw.k0.5.n70')


    ##################done
    # this was the one that hits bug, now ok - bug was negative bb values
    max_items = 50
    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp, weights=np.ones(fingerprint_length),
                                distance_function=NNSearch.distance_1_k,
                                distance_power=0.7, n_docs=max_items, use_visual_output1=False,
                                use_visual_output2=False, image_sets=None, self_reporting=None,
                                comparisons_to_make=None,
                                filename='gcfp.k0.7')

    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp, weights=np.ones(fingerprint_length),
                                distance_function=NNSearch.distance_1_k,
                                distance_power=1.1, n_docs=max_items, use_visual_output1=False,
                                use_visual_output2=False, image_sets=None, self_reporting=None,
                                comparisons_to_make=None,
                                filename='gcfp.k1.1')

    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp, weights=np.ones(fingerprint_length),
                                distance_function=NNSearch.distance_1_k,
                                distance_power=1.3, n_docs=max_items, use_visual_output1=False,
                                use_visual_output2=False, image_sets=None, self_reporting=None,
                                comparisons_to_make=None,
                                filename='gcfp.k1.3')



    ###################3


    # 9.6.15
    max_items = 30

    histogram_weights = 1.0 * np.ones(constants.histograms_length * 2)
    entropy_weights = np.ones(3)
    energy_weights = np.ones(3)
    weights = np.concatenate((energy_weights, entropy_weights, histogram_weights))
    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.regular_fp, weights=weights,
                                distance_function=NNSearch.distance_1_k, distance_power=0.5, n_docs=max_items,
                                filename='001.fp.n=30')

    max_items = 40
    histogram_weights = 1.0 * np.ones(constants.histograms_length * 2)
    entropy_weights = np.ones(3)
    energy_weights = np.ones(3)
    weights = np.concatenate((energy_weights, entropy_weights, histogram_weights))
    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.regular_fp, weights=weights,
                                distance_function=NNSearch.distance_1_k, distance_power=0.5, n_docs=max_items,
                                filename='002.fp.n=40')

    max_items = 50
    histogram_weights = 1.0 * np.ones(constants.histograms_length * 2)
    entropy_weights = np.ones(3)
    energy_weights = np.ones(3)
    weights = np.concatenate((energy_weights, entropy_weights, histogram_weights))
    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.regular_fp, weights=weights,
                                distance_function=NNSearch.distance_1_k, distance_power=0.5, n_docs=max_items,
                                filename='003.fp.n=50')

    max_items = 60
    histogram_weights = 1.0 * np.ones(constants.histograms_length * 2)
    entropy_weights = np.ones(3)
    energy_weights = np.ones(3)
    weights = np.concatenate((energy_weights, entropy_weights, histogram_weights))
    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.regular_fp, weights=weights,
                                distance_function=NNSearch.distance_1_k, distance_power=0.5, n_docs=max_items,
                                filename='004.fp.n=60')

    max_items = 50
    histogram_weights = 1.0 * np.ones(constants.histograms_length * 2)
    entropy_weights = np.ones(3)
    energy_weights = np.ones(3)
    weights = np.concatenate((energy_weights, entropy_weights, histogram_weights))
    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp, weights=weights,
                                distance_function=NNSearch.distance_1_k, distance_power=0.5, n_docs=max_items,
                                filename='005.gcfp')

    max_items = 70
    histogram_weights = np.ones(constants.histograms_length * 2)
    entropy_weights = np.ones(3)
    energy_weights = np.ones(3)
    weights = np.concatenate((energy_weights, entropy_weights, histogram_weights))
    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.regular_fp, weights=weights,
                                distance_function=NNSearch.distance_1_k, distance_power=0.5, n_docs=max_items,
                                filename='006.fp.n=70')

    max_items = 50
    histogram_weights = np.ones(constants.histograms_length * 2)
    entropy_weights = np.ones(3)
    energy_weights = np.ones(3)
    bwg_weights = np.ones(3)
    weights = np.concatenate((energy_weights, entropy_weights, histogram_weights, bwg_weights))
    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp_bw, weights=weights,
                                distance_function=NNSearch.distance_1_k, distance_power=0.5, n_docs=max_items,
                                filename='007.gcfpbw')

    histogram_weights = np.ones(constants.histograms_length * 2)
    entropy_weights = np.ones(3)
    energy_weights = np.ones(3)
    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp_histeq,
                                distance_function=NNSearch.distance_1_k, distance_power=0.5, n_docs=max_items,
                                filename='008.gcfp_histeq')

    histogram_weights = 1.5 * np.ones(constants.histograms_length * 2)
    entropy_weights = np.ones(3)
    energy_weights = np.ones(3)
    weights = np.concatenate((energy_weights, entropy_weights, histogram_weights))
    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp, weights=weights,
                                distance_function=NNSearch.distance_1_k, distance_power=0.5, n_docs=max_items,
                                filename='009.gcfp_histweighted')

    histogram_weights = np.ones(constants.histograms_length * 2)
    entropy_weights = 1.5 * np.ones(3)
    energy_weights = 1.5 * np.ones(3)
    bwg_weights = 1.5 * np.ones(3)
    weights = np.concatenate((energy_weights, entropy_weights, histogram_weights, bwg_weights))
    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp_bw, weights=weights,
                                distance_function=NNSearch.distance_1_k, distance_power=0.5, n_docs=max_items,
                                filename='010.gcfp.histdeweighted')
