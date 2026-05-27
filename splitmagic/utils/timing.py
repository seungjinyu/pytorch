import csv
import time


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def ms(self):
        return (time.perf_counter() - self.t0) * 1000


class CSVLogger:
    def __init__(self, path, header):
        self.path = path
        with open(self.path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    def write(self, row):
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)