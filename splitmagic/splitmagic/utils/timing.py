import csv
import os


class CSVLogger:
    def __init__(self, path, columns, append=False):
        self.path = path
        self.columns = columns

        file_exists = os.path.exists(path)
        file_has_content = file_exists and os.path.getsize(path) > 0

        mode = "a" if append else "w"

        with open(self.path, mode, newline="") as f:
            writer = csv.writer(f)

            if not append or not file_has_content:
                writer.writerow(columns)

    def write(self, row):
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)