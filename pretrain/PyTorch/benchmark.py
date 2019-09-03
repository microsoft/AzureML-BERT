import re
from datetime import datetime


def get_timestamp(text):
	datepattern = re.compile("\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}")
	matcher = datepattern.search(text)
	return datetime.strptime(matcher.group(0), '%m/%d/%Y %H:%M:%S')

def get_perf_metrics(filename):
	with open(filename) as f:
		datafile = f.readlines()
		throughput = 0
		epoch = 0
		time_diff=0
		for line in datafile:
			if 'Training epoch:' in line:
				start_time = get_timestamp(line)
				epoch += 1
			if 'Completed processing' in line:
				end_time = get_timestamp(line)
				time_diff = int((end_time-start_time).total_seconds())
				num_seq = [int(s) for s in line[int(line.find('Completed processing')):].split() if s.isdigit()][0]
				throughput += num_seq/time_diff
			if 'Validation Loss' in line:
				print('Validation loss:',float(line[int(line.find('is:'))+3:]))
		avg_throughput = throughput/float(epoch)
		print('Num epochs:', epoch, ' Average throughput:',avg_throughput, 'sequences/second')