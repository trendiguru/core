from multiprocessing import Process, Value, cpu_count
import ctypes


# Go here to determine output_ctype:
# https://docs.python.org/2/library/ctypes.html#fundamental-data-types

def map(f, inputs, num_processes=None, ouput_ctype=None):
    ouput_ctype = ouput_ctype or ctypes.c_bool
    num_processes = num_processes or int(0.75 * cpu_count())
    outputs = [Value(ctypes.c_bool) for i in xrange(num_processes)]
    chunked_inputs = [inputs[i:i + num_processes] for i in xrange(0, len(inputs), num_processes)]
    return_values = []
    for input_chunk in chunked_inputs:
        processes = [Process(target=worker, args=(f, input_chunk[i], outputs[i]))
                     for i in xrange(len(input_chunk))]
        [p.start() for p in processes]
        [p.join() for p in processes]
        [return_values.append(o.value) for o in outputs]
    return return_values

import numpy as np

def map2(f, inputs, num_processes=None, ouput_ctype=None):
    ouput_ctype = ouput_ctype or ctypes.c_bool
    num_processes = num_processes or int(0.75 * cpu_count())
    vars1=range(num_processes)
    vars2=range(num_processes)
    #doesnt work since i need two numpy arrays to be returned
    outputs = [Value(ctypes.c_bool) for i in xrange(num_processes)]
    #doesnt work since its pass by value
    outputs = [[vars1[i],vars2[i]] for i in xrange(num_processes)]
    #doesnt work since 'this type has no size'
#    outputs = [[Value(np.array),Value(np.array)] for i in xrange(num_processes)]
    chunked_inputs = [inputs[i:i + num_processes] for i in xrange(0, len(inputs), num_processes)]
    return_values = []
    for input_chunk in chunked_inputs:
        processes = [Process(target=worker2, args=(f, input_chunk[i], outputs[i]))
                     for i in xrange(len(input_chunk))]
        [p.start() for p in processes]
        [p.join() for p in processes]
        [return_values.append(o) for o in outputs]
    return return_values

def worker(f, args, output):
    output.value = f(*args)

def worker2(f, args, output):
#    output[0],output[1] = f(args)
#    output.value = f(args)
    output = f(args)
    print('output {} args {}'.format(output,args))
    return output

def check_small(input1, input2):
    print "Will wait then work"
    time.sleep(1)
    return bool(input1 * input2 < 0.25)


def run_test():
    import random
    start = time.time()
    image_and_page_urls = [(random.random(), random.random()) for i in xrange(20)]

    outs = map(check_small, image_and_page_urls)

    print "outs: "
    print outs

    print "All set, please come again!"
    print time.time() - start


if __name__ == '__main__':
    import time

    tstart = time.time()
    run_test()
    print "Run time: " + str(time.time() - tstart)
