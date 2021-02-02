import time
import argparse
import pynvml


class Device(object):
    class Status:
        INIT = "INIT"
        DETECTING = "DETECTING"
        STOP = "STOP"

    start_detecting_mem_threshold = 32 * 1024 * 1024

    def __init__(self, handle):
        self.handle = handle
        self.status = self.Status.INIT
        self.max_mem_usage = 0

    def update(self):
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        if self.status == self.Status.INIT:
            if info.used > self.start_detecting_mem_threshold:
                self.status = self.Status.DETECTING
        elif self.status == self.Status.DETECTING:
            if info.used < self.start_detecting_mem_threshold:
                self.status = self.Status.STOP
                return False
            else:
                self.max_mem_usage = max(self.max_mem_usage, info.used)
        elif self.status == self.Status.STOP:
            raise ValueError("detecting is stop")
        else:
            raise ValueError("invalid status")

        return True


def main():
    parser = argparse.ArgumentParser(description="collect GPU device memory usage")
    parser.add_argument("-g", type=int, default=1, help="number of gpu devices")
    parser.add_argument("-n", type=float, default=1, help="metrics rate")
    args = parser.parse_args()

    pynvml.nvmlInit()
    n_gpus = args.g
    devices = [Device(pynvml.nvmlDeviceGetHandleByIndex(i)) for i in range(n_gpus)]

    running = True
    while running:
        time.sleep(args.n)
        running = False
        for device in devices:
            running |= device.update()

    pynvml.nvmlShutdown()
    for i, device in enumerate(devices):
        max_mem_usage_mbytes = device.max_mem_usage / 1024 / 1024
        print(f"gpt{i} max memory usage: {max_mem_usage_mbytes:.2f}M")


if __name__ == "__main__":
    main()
