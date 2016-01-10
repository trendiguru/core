__author__ = 'yonatan'
import time

import numpy as np

from new_finger_print import spaciograms_distance_rating


def distance_function_nate(entry, target_dict, index):
    a = time.time()
    dist = spaciograms_distance_rating(np.asarray(entry["specio"][index]), target_dict["specio"][index])
    b = time.time()
    print ("specio time = %s" % str(b - a))

    return dist


def stage_one(target_dict, entries):
    start_time = time.time()
    # list of tuples with (entry,distance). Initialize with first n distance values
    nearest_n = []
    farthest_nearest = 20000
    for i, entry in enumerate(entries):
        if i < 1000:
            d = distance_function_nate(entry, target_dict)
            nearest_n.append((entry, d))
        else:
            if i == 100:
                # sort by distance
                nearest_n.sort(key=lambda tup: tup[1])
                # last item in the list (index -1, go python!)
                farthest_nearest = nearest_n[-1][1]

            # Loop through remaining entries, if one of them is better, insert it in the correct location and remove last item
            d = distance_function_nate(entry, target_dict)
            if d < farthest_nearest:
                insert_at = 98
                while d < nearest_n[insert_at][1]:
                    insert_at -= 1
                    if insert_at == -1:
                        break
                nearest_n.insert(insert_at + 1, (entry, d))
                nearest_n.pop()
                farthest_nearest = nearest_n[-1][1]
    end_time = time.time()
    total_time = end_time - start_time
    print ("total time = %s" % (str(total_time)))
    return nearest_n
