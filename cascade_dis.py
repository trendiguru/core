__author__ = 'yonatan'
import time

from new_finger_print import spaciograms_distance_rating


def distance_function_nate(entry, target_dict, rank):
    # a = time.time()
    if rank == 1:
        dist = spaciograms_distance_rating(entry["sp_one"], target_dict["sp_one"], rank)
    elif rank == 2:
        dist = spaciograms_distance_rating(entry["sp_two"], target_dict["sp_two"], rank)
    elif rank == 3:
        dist = spaciograms_distance_rating(entry["specio"], target_dict["specio"], rank)
    # b = time.time()
    # print ("specio time = %s" % str(b - a))

    return dist


def stage_one(target_dict, entries, rank, stopme):
    start_time = time.time()
    # list of tuples with (entry,distance). Initialize with first n distance values
    nearest_n = []
    farthest_nearest = 20000
    i = 0
    for entry in entries:
        if (i % 1000) == 0:
            print (i)
        # print ("boom")
        if i < stopme:
            d = distance_function_nate(entry, target_dict, rank)
            nearest_n.append((entry, d))
        else:
            if i == stopme:
                # sort by distance
                nearest_n.sort(key=lambda tup: tup[1])
                # last item in the list (index -1, go python!)
                farthest_nearest = nearest_n[-1][1]

            # Loop through remaining entries, if one of them is better, insert it in the correct location and remove last item
            d = distance_function_nate(entry, target_dict, rank)

            if d < farthest_nearest:
                insert_at = stopme - 2
                while d < nearest_n[insert_at][1]:
                    insert_at -= 1
                    if insert_at == -1:
                        break
                nearest_n.insert(insert_at + 1, (entry, d))
                nearest_n.pop()
                farthest_nearest = nearest_n[-1][1]
        i += 1
    end_time = time.time()
    total_time = end_time - start_time
    print ("total time = %s" % (str(total_time)))
    id_list = [item['id'] for item in nearest_n]
    return id_list, nearest_n
