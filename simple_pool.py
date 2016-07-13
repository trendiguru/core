import os
import sys
import time
import gevent
from gevent import server, event, socket
from multiprocessing import Process, Value, current_process, cpu_count
import ctypes
import random

# Go here to determine output_ctype:
# https://docs.python.org/2/library/ctypes.html#fundamental-data-types
def run_parallel(f, inputs, num_processes=None, ouput_ctype=None):
	ouput_ctype = ouput_ctype or ctypes.c_bool
	num_processes = num_processes or int(0.75 * cpu_count())
	outputs = [Value(ctypes.c_bool) for i in xrange(num_processes)]
	chunked_inputs = [inputs[i:i+num_processes] for i in xrange(0, len(inputs), num_processes)]
	return_values = []
	for input_chunk in chunked_inputs:
		processes = [Process(target=worker, args=(f, input_chunk[i], outputs[i])) 
						for i in xrange(len(input_chunk))]
		[p.start() for p in processes]
		[p.join() for p in processes]
		[return_values.append(o.value) for o in outputs]
	return return_values

def worker(f, args, output):
	output.value = f(*args)


def check_small(input1, input2):
	print "Will wait then work"
	time.sleep(1)
	return bool(input1*input2 < 0.25)



def run_test():
	start = time.time()
	image_and_page_urls = [(random.random(), random.random()) for i in xrange(20)]

	outs = run_parallel(check_small, image_and_page_urls)

	print "outs: "
	print outs
	
	print "All set, please come again!"
	print time.time() - start


	

if __name__ == '__main__':
	tstart = time.time()
	run_test()
	print "Run time: " + str(time.time()-tstart)
