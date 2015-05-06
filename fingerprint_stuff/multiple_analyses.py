__author__ = 'jeremy'
import numpy as np

import rate_fp
import fingerprint_core as fp_core
import NNSearch
import constants

################
# started 30.4
####################
fingerprint_length = constants.fingerprint_length

############ added 3.5.15
max_items = 5
rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp_bw, weights=np.ones(fingerprint_length),
                            distance_function=NNSearch.distance_1_k,
                            distance_power=0.5, n_docs=max_items, use_visual_output1=False,
                            use_visual_output2=False, image_sets=None, self_reporting=None, comparisons_to_make=None,
                            filename='gcfpbw.k0.5')

max_items = 50
rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp_bw, weights=np.ones(fingerprint_length),
                            distance_function=NNSearch.distance_1_k,
                            distance_power=0.5, n_docs=max_items, use_visual_output1=False,
                            use_visual_output2=False, image_sets=None, self_reporting=None, comparisons_to_make=None,
                            filename='gcfpbw.k0.5')

max_items = 70
rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp_bw, weights=np.ones(fingerprint_length),
                            distance_function=NNSearch.distance_1_k,
                            distance_power=0.5, n_docs=max_items, use_visual_output1=False,
                            use_visual_output2=False, image_sets=None, self_reporting=None, comparisons_to_make=None,
                            filename='gcfpbw.k0.5.n70')


##################done
# this was the one that hits bug
max_items = 70
rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp, weights=np.ones(fingerprint_length),
                            distance_function=NNSearch.distance_1_k,
                            distance_power=0.5, n_docs=max_items, use_visual_output1=False,
                            use_visual_output2=False, image_sets=None, self_reporting=None, comparisons_to_make=None,
                            filename='gcfp_n70')

max_items = 50
rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp, weights=np.ones(fingerprint_length),
                            distance_function=NNSearch.distance_1_k,
                            distance_power=0.7, n_docs=max_items, use_visual_output1=False,
                            use_visual_output2=False, image_sets=None, self_reporting=None, comparisons_to_make=None,
                            filename='gcfp.k0.7')

rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp, weights=np.ones(fingerprint_length),
                            distance_function=NNSearch.distance_1_k,
                            distance_power=1.1, n_docs=max_items, use_visual_output1=False,
                            use_visual_output2=False, image_sets=None, self_reporting=None, comparisons_to_make=None,
                            filename='gcfp.k1.1')

rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp, weights=np.ones(fingerprint_length),
                            distance_function=NNSearch.distance_1_k,
                            distance_power=1.3, n_docs=max_items, use_visual_output1=False,
                            use_visual_output2=False, image_sets=None, self_reporting=None, comparisons_to_make=None,
                            filename='gcfp.k1.3')

max_items = 60
rate_fp.analyze_fingerprint(fingerprint_function=fp_core.regular_fp, weights=np.ones(fingerprint_length),
                            distance_function=NNSearch.distance_1_k,
                            distance_power=0.5, n_docs=max_items, use_visual_output1=False,
                            use_visual_output2=False, image_sets=None, self_reporting=None, comparisons_to_make=None,
                            filename='fp_n60')


# max_items = 20
rate_fp.analyze_fingerprint(fingerprint_function=fp_core.regular_fp, weights=np.ones(fingerprint_length),
                            distance_function=NNSearch.distance_1_k,
                            distance_power=0.5, n_docs=max_items, use_visual_output1=False,
                            use_visual_output2=False, image_sets=None, self_reporting=None, comparisons_to_make=None,
                            filename='fp_n20')

max_items = 40
rate_fp.analyze_fingerprint(fingerprint_function=fp_core.regular_fp, weights=np.ones(fingerprint_length),
                            distance_function=NNSearch.distance_1_k,
                            distance_power=0.5, n_docs=max_items, use_visual_output1=False,
                            use_visual_output2=False, image_sets=None, self_reporting=None, comparisons_to_make=None,
                            filename='fp_n40')

max_items = 50
rate_fp.analyze_fingerprint(fingerprint_function=fp_core.regular_fp, weights=np.ones(fingerprint_length),
                            distance_function=NNSearch.distance_1_k,
                            distance_power=0.5, n_docs=max_items, use_visual_output1=False,
                            use_visual_output2=False, image_sets=None, self_reporting=None, comparisons_to_make=None,
                            filename='fp_n50')

max_items = 50
rate_fp.analyze_fingerprint(fingerprint_function=fp_core.regular_fp, weights=np.ones(fingerprint_length),
                            distance_function=NNSearch.distance_1_k,
                            distance_power=0.5, n_docs=max_items, use_visual_output1=False,
                            use_visual_output2=False, image_sets=None, self_reporting=None, comparisons_to_make=None,
                            filename='fp_n50')

max_items = 50
rate_fp.analyze_fingerprint(fingerprint_function=fp_core.regular_fp, weights=np.ones(fingerprint_length),
                            distance_function=NNSearch.distance_1_k,
                            distance_power=0.5, n_docs=max_items, use_visual_output1=False,
                            use_visual_output2=False, image_sets=None, self_reporting=None, comparisons_to_make=None,
                            filename='fp_n50')


#####runs into bug. nope actually ok
max_items = 80
rate_fp.analyze_fingerprint(fingerprint_function=fp_core.regular_fp, weights=np.ones(fingerprint_length),
                            distance_function=NNSearch.distance_1_k,
                            distance_power=0.5, n_docs=max_items, use_visual_output1=False,
                            use_visual_output2=False, image_sets=None, self_reporting=None, comparisons_to_make=None,
                            filename='fp_n80')



######started  5.5
max_items = 50
rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp, weights=np.ones(fingerprint_length),
                            distance_function=NNSearch.distance_1_k,
                            distance_power=0.5, n_docs=max_items, use_visual_output1=False,
                            use_visual_output2=False, image_sets=None, self_reporting=None, comparisons_to_make=None,
                            filename='gcfp_n50')


###################3

n_groups = [20, 40, 50, 60, 70, 80]
goodnesses = [0, .121, .109, .127, .115, .1]
g_error = [0, .089, .0803, .07, 067, .06]