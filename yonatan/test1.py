#!/usr/bin/env python

import yonatan_constants

dict = yonatan_constants.attribute_type_dict


for key, value in dict.iteritems():

    counter = 0

    if value[1] == "fabric":
        counter += 1
        print value[0]

print "\ncounter: {0}".format(counter)

