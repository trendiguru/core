__author__ = 'jeremy'
import numpy as np

import rate_fp
import fingerprint_core as fp_core
import NNSearch
import constants

################
# started 30.4
####################
max_items = 50
fingerprint_length = constants.fingerprint_length

rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp, weights=np.ones(fingerprint_length),
                            distance_function=NNSearch.distance_1_k,
                            distance_power=0.5, n_docs=max_items, use_visual_output1=False,
                            use_visual_output2=False, image_sets=None, self_reporting=None, comparisons_to_make=None,
                            filename='gcfp_fp50')

max_items = 70
rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp, weights=np.ones(fingerprint_length),
                            distance_function=NNSearch.distance_1_k,
                            distance_power=0.5, n_docs=max_items, use_visual_output1=False,
                            use_visual_output2=False, image_sets=None, self_reporting=None, comparisons_to_make=None,
                            filename='gcfp_n70')

max_items = 90
rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp, weights=np.ones(fingerprint_length),
                            distance_function=NNSearch.distance_1_k,
                            distance_power=0.5, n_docs=max_items, use_visual_output1=False,
                            use_visual_output2=False, image_sets=None, self_reporting=None, comparisons_to_make=None,
                            filename='gcfp_n90')

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

############ added 3.5.15
rate_fp.analyze_fingerprint(fingerprint_function=fp_core.gc_and_fp_bw, weights=np.ones(fingerprint_length),
                            distance_function=NNSearch.distance_1_k,
                            distance_power=0.5, n_docs=max_items, use_visual_output1=False,
                            use_visual_output2=False, image_sets=None, self_reporting=None, comparisons_to_make=None,
                            filename='gcfpbw.k0.5')

###################3

