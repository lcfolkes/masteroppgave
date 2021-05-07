import multiprocessing
import concurrent.futures
import bs4 as bs
import random
import requests
import string
import time

def do_something_print(seconds):
	print(f'Sleeping {seconds} second...')
	time.sleep(seconds)
	print('Done Sleeping...')

def do_something_return(seconds):
	print(f'Sleeping {seconds} second...')
	time.sleep(seconds)
	return f'Done Sleeping...{seconds}'

if __name__ == '__main__':
	start = time.perf_counter()

	'''
	processes = []
	for i in range(10):
		p = multiprocessing.Process(target=do_something_print, args=(1.5,))
		p.start()
		processes.append(p)

	for process in processes:
		process.join()
	'''

	with concurrent.futures.ProcessPoolExecutor() as executor:
		secs = [5, 4, 3, 2, 1]
		results = executor.map(do_something_return, secs)
		for result in results:
			print(result)

	finish = time.perf_counter()
	print(f"Finished in {round(finish-start, 2)} seconds(s)")
