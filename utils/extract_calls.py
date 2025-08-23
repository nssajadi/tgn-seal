import scipy.io
import csv
import math
from datetime import datetime, timedelta


class Call:
    def __init__(self, node_src, node_dst, date, event, duration):
        self.node_src = node_src
        self.node_dst = node_dst
        self.date = datetime.fromordinal(int(date)) + timedelta(days=date % 1) - timedelta(days=366)
        self.event = event
        self.duration = duration


class DataSet:
    def __init__(self, _data_path):
        self.data_path = _data_path
        self.data = scipy.io.loadmat(_data_path)
        self.comm = self.data['s']['comm'][0]
        self.hashed_number = self.data['s']['my_hashedNumber'][0]


def extract_calls(_dataset):
    call_records = []
    comm = _dataset.comm
    hashed_number = _dataset.hashed_number
    for i in range(comm.size):
        for j in range(comm[i].size):
            c = comm[i][0][j]
            if c[3][0] == 'Voice call' and c[4][0] != 'Missed' and c[5][0][0] != 0:
                try:
                    if c[4][0] == 'Outgoing':
                        call_records.append(Call(hashed_number[i][0][0], c[6][0][0], c[0][0][0], c[1][0][0], c[5][0][0]))
                    else:
                        call_records.append(Call(c[6][0][0], hashed_number[i][0][0], c[0][0][0], c[1][0][0], c[5][0][0]))
                except Exception as e:
                    pass

    return call_records


def write_call_records_to_file(call_records):
    header = ['node src', 'node dst', 'timestamp', 'duration']
    with open('data/calls.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for c in call_records:
            if not (math.isnan(c.node_src) or math.isnan(c.node_dst)):
                data = [c.node_src, c.node_dst, int(datetime.timestamp(c.date)), c.duration]
                writer.writerow(data)


data_path = './data/realitymining.mat'
dataset = DataSet(data_path)

calls = extract_calls(dataset)
calls = sorted(calls, key=lambda x: int(datetime.timestamp(x.date)))
pairs = list(set([(c.node_src, c.node_dst) for c in calls]))

write_call_records_to_file(calls)
