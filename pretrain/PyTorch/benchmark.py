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
		epoch = 1
		time_diff=0
		num_seq=0
		for line in datafile:
			if 'Training epoch:' in line:
				start_time = get_timestamp(line)

				if epoch == 1:
					training_start_time = start_time
				epoch += 1
			if 'Completed processing' in line:
				end_time = get_timestamp(line)
				time_diff += int((end_time-start_time).total_seconds())
				num_seq += [int(s) for s in line[int(line.find('Completed processing')):].split() if s.isdigit()][0]
				throughput = num_seq/time_diff
				#print(throughput)
			if 'Validation Loss' in line:
				valid_loss = float(line[int(line.find('is:'))+3:])
		avg_throughput = (num_seq/time_diff)
		total_training_time = end_time-training_start_time
		d = datetime(1,1,1) + total_training_time

		print('Num epochs:', epoch)
		print('Total time to train:', d.day-1,'days,', d.hour ,'hours')
		print('Average throughput:',avg_throughput, 'sequences/second')
		print('Final Validation Loss:', valid_loss)
