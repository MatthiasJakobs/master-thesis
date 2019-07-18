import time
import os

class Timer():
    def __init__(self, name, experiment_name):
        self.name = name
        self.experiment_name = experiment_name

        self.min = 1e20
        self.max = 1e-20

        self.sum = 0
        self.count = 0

        self.running = False

    def start(self):
        if not self.running:
            self.start_time = time.time()
            self.running = True

    def stop(self):
        if self.running:
            self.stop_time = time.time()

            interval_sec = self.stop_time - self.start_time

            self.sum = self.sum + interval_sec
            self.count = self.count + 1

            if interval_sec < self.min:
                self.min = interval_sec

            if interval_sec > self.max:
                self.max = interval_sec

            self.running = False

    def save(self):
        out_path = "experiments/{}/timing_{}.txt".format(self.experiment_name, self.name)

        with open(out_path, "w+") as out_file:
            out_file.write("sum,mean,max,min\n")
            out_file.write("{},{},{},{}\n".format(self.sum, float(self.sum) / self.count, self.max, self.min))


