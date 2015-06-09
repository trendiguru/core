__author__ = 'jeremy'
import numpy as np

import rate_fp
import fingerprint_core as fp_core
import NNSearch
import constants

fingerprint_length = constants.fingerprint_length

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

#########
# max_items = 50
# DONE
#########
if (0):
    max_items = 50
    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.regular_fp, weights=weights,
                                distance_function=NNSearch.distance_1_k, distance_power=0.5, n_docs=max_items,
                                filename='002.fp.n=50')

    max_items = 70
    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.regular_fp, weights=weights,
                                distance_function=NNSearch.distance_1_k, distance_power=0.5, n_docs=max_items,
                                filename='001.fp.n=70')

    histogram_weights = np.ones(constants.histograms_length * 2)
    entropy_weights = np.ones(3)
    energy_weights = np.ones(3)
    bwg_weights = np.ones(3)
    weights = np.concatenate((energy_weights, entropy_weights, histogram_weights, bwg_weights))
    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp_bw, weights=weights,
                                distance_function=NNSearch.distance_1_k, distance_power=0.5, n_docs=max_items,
                                filename='001.gcfpbw')

    histogram_weights = 1.5 * np.ones(constants.histograms_length * 2)
    entropy_weights = np.ones(3)
    energy_weights = np.ones(3)
    weights = np.concatenate((energy_weights, entropy_weights, histogram_weights))
    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp, weights=weights,
                                distance_function=NNSearch.distance_1_k, distance_power=0.5, n_docs=max_items,
                                filename='002.gcfp')




    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp_histeq,
                                distance_function=NNSearch.distance_1_k, distance_power=0.5, n_docs=max_items,
                                filename='gcfp_histeq')

    histogram_weights = np.ones(constants.histograms_length * 2)
    entropy_weights = 3 * np.ones(3)
    energy_weights = 3 * np.ones(3)
    bwg_weights = 3 * np.ones(3)
    weights = np.concatenate((energy_weights, entropy_weights, histogram_weights, bwg_weights))
    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp_bw, weights=weights,
                                distance_function=NNSearch.distance_1_k, distance_power=0.5, n_docs=max_items,
                                filename='gcfp.weights3')

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
    max_items = 70
    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp, weights=np.ones(fingerprint_length),
                                distance_function=NNSearch.distance_1_k,
                                distance_power=0.5, n_docs=max_items, use_visual_output1=False,
                                use_visual_output2=False, image_sets=None, self_reporting=None,
                                comparisons_to_make=None,
                                filename='gcfp_n70')

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

    max_items = 60
    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.regular_fp, weights=np.ones(fingerprint_length),
                                distance_function=NNSearch.distance_1_k,
                                distance_power=0.5, n_docs=max_items, use_visual_output1=False,
                                use_visual_output2=False, image_sets=None, self_reporting=None,
                                comparisons_to_make=None,
                                filename='fp_n60')


    # max_items = 20
    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.regular_fp, weights=np.ones(fingerprint_length),
                                distance_function=NNSearch.distance_1_k,
                                distance_power=0.5, n_docs=max_items, use_visual_output1=False,
                                use_visual_output2=False, image_sets=None, self_reporting=None,
                                comparisons_to_make=None,
                                filename='fp_n20')

    max_items = 40
    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.regular_fp, weights=np.ones(fingerprint_length),
                                distance_function=NNSearch.distance_1_k,
                                distance_power=0.5, n_docs=max_items, use_visual_output1=False,
                                use_visual_output2=False, image_sets=None, self_reporting=None,
                                comparisons_to_make=None,
                                filename='fp_n40')

    max_items = 50
    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.regular_fp, weights=np.ones(fingerprint_length),
                                distance_function=NNSearch.distance_1_k,
                                distance_power=0.5, n_docs=max_items, use_visual_output1=False,
                                use_visual_output2=False, image_sets=None, self_reporting=None,
                                comparisons_to_make=None,
                                filename='fp_n50')

    max_items = 50
    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.regular_fp, weights=np.ones(fingerprint_length),
                                distance_function=NNSearch.distance_1_k,
                                distance_power=0.5, n_docs=max_items, use_visual_output1=False,
                                use_visual_output2=False, image_sets=None, self_reporting=None,
                                comparisons_to_make=None,
                                filename='fp_n50')

    max_items = 50
    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.regular_fp, weights=np.ones(fingerprint_length),
                                distance_function=NNSearch.distance_1_k,
                                distance_power=0.5, n_docs=max_items, use_visual_output1=False,
                                use_visual_output2=False, image_sets=None, self_reporting=None,
                                comparisons_to_make=None,
                                filename='fp_n50')


    #####runs into bug. nope actually ok
    max_items = 80
    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.regular_fp, weights=np.ones(fingerprint_length),
                                distance_function=NNSearch.distance_1_k,
                                distance_power=0.5, n_docs=max_items, use_visual_output1=False,
                                use_visual_output2=False, image_sets=None, self_reporting=None,
                                comparisons_to_make=None,
                                filename='fp_n80')



    ######started  5.5
    max_items = 50
    rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp, weights=np.ones(fingerprint_length),
                                distance_function=NNSearch.distance_1_k,
                                distance_power=0.5, n_docs=max_items, use_visual_output1=False,
                                use_visual_output2=False, image_sets=None, self_reporting=None,
                                comparisons_to_make=None,
                                filename='gcfp_n50')


    ###################3

n_groups = [20, 40, 50, 60, 70, 80]
goodnesses = [0, .121, .109, .127, .115, .1]
g_error = [0, .089, .0803, .07, 067, .06]
results = [{'fingerprint_function': 'gcbw_n50', 'goodness': 0.24, 'goodness_error': 0.06},
           {'fingerprint_function': 'gcbw', 'goodness': 0.21, 'goodness_error': 0.23},
           {'fingerprint_function': 'gcbw', 'goodness': 0.21, 'goodness_error': 0.23},
           {'fingerprint_function': 'gcbw', 'goodness': 0.21, 'goodness_error': 0.23}]

# gcfpbw
res = []
res.append({
    "cross_report": {
        "average_unweighted": 1.05027,
        "average_weighted": 1.05027,
        "distance_function": "<function distance_1_k at 0x7f3175330398>",
        "distance_power": 0.5,
        "error_cumulative": 0.03941,
        "fingerprint_function": "<function gc_and_fp_bw at 0x7f3174f300c8>",
        "n_groups": 50,
        "timestamp": "2015-05-06 16:44",
        "tot_images": 634
    },
    "goodness": 0.23512669936531733,
    "goodness_error": 0.058084978075840905,
    "self_report": {
        "average_unweighted": 0.80439,
        "average_weighted": 0.80623,
        "distance_function": "<function distance_1_k at 0x7f3175330398>",
        "distance_power": 0.5,
        "error_cumulative": 0.04593,
        "fingerprint_function": "<function gc_and_fp_bw at 0x7f3174f300c8>",
        "n_groups": 50,
        "timestamp": "2015-05-06 16:05",
        "tot_images": 634
    }
})

res.append({
    "cross_report": {
        "average_unweighted": 0.90665,
        "average_weighted": 0.90665,
        "distance_function": "<function distance_1_k at 0x7f3175330398>",
        "distance_power": 0.5,
        "error_cumulative": 0.04531,
        "fingerprint_function": "<function regular_fp at 0x7f3174f30140>",
        "n_groups": 60,
        "timestamp": "2015-05-07 00:44",
        "tot_images": 786
    },
    "goodness": 0.12320875718708539,
    "goodness_error": 0.07328553512791537,
    "self_report": {
        "average_unweighted": 0.79128,
        "average_weighted": 0.79297,
        "distance_function": "<function distance_1_k at 0x7f3175330398>",
        "distance_power": 0.5,
        "error_cumulative": 0.04805,
        "fingerprint_function": "<function regular_fp at 0x7f3174f30140>",
        "n_groups": 60,
        "timestamp": "2015-05-07 00:44",
        "tot_images": 786
    }})
res.append({
    "cross_report": {
        "average_unweighted": 1.05027,
        "average_weighted": 1.05027,
        "distance_function": "<function distance_1_k at 0x7f3175330398>",
        "distance_power": 0.5,
        "error_cumulative": 0.03941,
        "fingerprint_function": "<function gc_and_fp_bw at 0x7f3174f300c8>",
        "n_groups": 50,
        "timestamp": "2015-05-06 16:44",
        "tot_images": 634
    },
    "goodness": 0.23512669936531733,
    "goodness_error": 0.058084978075840905,
    "self_report": {
        "average_unweighted": 0.80439,
        "average_weighted": 0.80623,
        "distance_function": "<function distance_1_k at 0x7f3175330398>",
        "distance_power": 0.5,
        "error_cumulative": 0.04593,
        "fingerprint_function": "<function gc_and_fp_bw at 0x7f3174f300c8>",
        "n_groups": 50,
        "timestamp": "2015-05-06 16:05",
        "tot_images": 634
    }})

res.append({
    "cross_report": {
        "average_unweighted": 1.04785,
        "average_weighted": 1.04785,
        "distance_function": "<function distance_1_k at 0x7f3175330398>",
        "distance_power": 0.5,
        "error_cumulative": 0.0346,
        "fingerprint_function": "<function gc_and_fp_bw at 0x7f3174f300c8>",
        "n_groups": 70,
        "timestamp": "2015-05-06 18:21",
        "tot_images": 919
    },
    "goodness": 0.21746929895707104,
    "goodness_error": 0.05114521561329079,
    "self_report": {
        "average_unweighted": 0.81795,
        "average_weighted": 0.8201,
        "distance_function": "<function distance_1_k at 0x7f3175330398>",
        "distance_power": 0.5,
        "error_cumulative": 0.04024,
        "fingerprint_function": "<function gc_and_fp_bw at 0x7f3174f300c8>",
        "n_groups": 70,
        "timestamp": "2015-05-06 17:40",
        "tot_images": 919
    }
})


