from collections import defaultdict
import contextlib
import time

class PerfLogger():
    def __init__(self, print=True):
        self.time_measure_result = defaultdict(list)
        self.print = print

    @contextlib.contextmanager
    def time_measure(self, log_name):
        start_time = time.time()
        yield
        end_time = time.time()
        if self.print:
            print(f"{log_name}: {end_time - start_time}")
        # self,time_measure_result[log_name].append((end_time - start_time) * 1000)
        # if log_name not in manager_dict:
        #     manager_dict[log_name] = manager.list()
        # manager_dict[log_name].append((end_time - start_time) * 1000)