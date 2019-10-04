#!/usr/bin/env python3

import os
from subprocess import Popen, PIPE, STDOUT


def find_empty_gpu():
	'''
	Pick an empty GPU: Query using nvidia-smi, parse output and
	return the first available GPU that has no process associated
	'''
	p = Popen(['nvidia-smi', '-q'], stdout=PIPE, stderr=PIPE)
	(stdout, stderr) = p.communicate()
	stdout = stdout.decode("utf-8")
	gpu_number = ''
	for line in stdout.split('\n'):
		# Parse output, format is 'key : value'
		if line.find(':') < 0:
			continue
		(key, value) = [s.strip() for s in line.split(':', maxsplit=1)]
		if key == 'Minor Number':
			gpu_number = value
		if key == 'Processes' and value == 'None' and gpu_number:
			return gpu_number
	return None


def select_empty_gpu(no_gpu_error=False):
	''' Find and select the first empty GPU '''
	gpu_number = find_empty_gpu()
	if not gpu_number:
		if no_gpu_error:
			raise "Could not find an empty GPU"
		else:
			print("WARNING: No available GPU found")
	else:
		os.environ['CUDA_VISIBLE_DEVICES'] = gpu_number
