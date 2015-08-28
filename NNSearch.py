# TODO - write test functions for main functions here

__author__ = 'jeremy'
import logging
import numpy as np
import constants


K = constants.K  # .5 is the same as Euclidean
FP_KEY = "fingerPrintVector"


def distance_1_k(fp1, fp2, k=K):
    """This calculates distance between to arrays. When k = .5 this is the same as Euclidean."""
    if fp1 is not None and fp2 is not None:
        f12 = np.abs(np.array(fp1) - np.array(fp2))
        f12_p = np.power(f12, 1 / k)
        return np.power(np.sum(f12_p), k)
    else:
        print("null fingerprint sent to distance_1_k ")
        logging.warning("null fingerprint sent to distance_1_k ")
        return None


def find_n_nearest_neighbors(target_dict, entries, number_of_matches, distance_function=distance_1_k, fp_key=FP_KEY):
    # list of tuples with (entry,distance). Initialize with first n distance values
    nearest_n = [(entries[i], distance_function(entries[i][fp_key], target_dict[fp_key]))
                 for i in range(0, number_of_matches)]
    # sort by distance
    nearest_n.sort(key=lambda tup: tup[1])
    # last item in the list (index -1, go python!)
    farthest_nearest = nearest_n[-1][1]

    # Loop through remaining entries, if one of them is better, insert it in the correct location and remove last item
    for entry in entries[number_of_matches:]:
        d = distance_function(entry[fp_key], target_dict[fp_key], K)
        if d < farthest_nearest:
            insert_at = number_of_matches-2
            while d < nearest_n[insert_at][1]:
                insert_at -= 1
                if insert_at == -1:
                    break
            nearest_n.insert(insert_at + 1, (entry, d))
            nearest_n.pop()
            farthest_nearest = nearest_n[-1][1]
    return nearest_n
