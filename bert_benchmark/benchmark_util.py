import time
import numpy as np


class StopWatch:
    def __init__(self):
        pass

    def start(self):
        self.start_time = time.time()
        self.last_split = self.start_time

    def split(self):
        now = time.time()
        duration = now - self.last_split
        self.last_split = now
        return duration

    def stop(self):
        self.stop_time = time.time()

    def duration(self):
        return self.stop_time - self.start_time


class CNNSpeedometer:
    def __init__(self):
        self.watch = StopWatch()
        self.throughoutput_list = []

    def speedometer_cb(
        self, step, total_batch_size, skip_iter_num, iter_num, loss_print_every_n_iter
    ):
        def callback(train_loss):
            if step < skip_iter_num:
                print(
                    "Runing warm up for {}/{} iterations.".format(
                        step + 1, skip_iter_num
                    )
                )
                if (step + 1) >= skip_iter_num:
                    self.watch.start()
                    print("Start trainning.")
            else:
                train_step = step - skip_iter_num

                if (train_step + 1) % loss_print_every_n_iter == 0:
                    loss = train_loss.mean()
                    avg_elapse_time_per_iter = (
                        self.watch.split() / loss_print_every_n_iter
                    )
                    images_per_sec = total_batch_size / avg_elapse_time_per_iter
                    print(
                        "iter {}, loss: {:.3f}, speed: {:.3f}(sec/batch), {:.3f}(images/sec)".format(
                            train_step, loss, avg_elapse_time_per_iter, images_per_sec
                        )
                    )
                    self.throughoutput_list.append(images_per_sec)

                if (train_step + 1) == iter_num:
                    self.watch.stop()
                    print("-".ljust(66, "-"))
                    print(
                        "average speed: {:.3f}(images/sec)".format(
                            np.mean(self.throughoutput_list)
                        )
                    )
                    print("-".ljust(66, "-"))

        return callback


class BERTSpeedometer:
    def __init__(self):
        self.watch = StopWatch()
        self.throughoutput_list = []

    def speedometer_cb(
        self, step, total_batch_size, skip_iter_num, iter_num, loss_print_every_n_iter
    ):
        def callback(train_loss):
            if step < skip_iter_num:
                print(
                    "Runing warm up for {}/{} iterations.".format(
                        step + 1, skip_iter_num
                    )
                )
                if (step + 1) >= skip_iter_num:
                    self.watch.start()
                    print("Start trainning.")
            else:
                train_step = step - skip_iter_num

                if (train_step + 1) % loss_print_every_n_iter == 0:
                    total_loss = train_loss[0].mean()
                    mlm_loss = train_loss[1].mean()
                    nsp_loss = train_loss[2].mean()

                    avg_elapse_time_per_iter = (
                        self.watch.split() / loss_print_every_n_iter
                    )
                    sentences_per_sec = total_batch_size / avg_elapse_time_per_iter
                    print(
                        "iter {}, total_loss: {:.3f}, mlm_loss: {:.3f}, nsp_loss: {:.3f}, speed: {:.3f}(sec/batch), {:.3f}(sentences/sec)".format(
                            train_step,
                            total_loss,
                            mlm_loss,
                            nsp_loss,
                            avg_elapse_time_per_iter,
                            sentences_per_sec,
                        )
                    )
                    self.throughoutput_list.append(sentences_per_sec)

                if (train_step + 1) == iter_num:
                    self.watch.stop()
                    print("-".ljust(66, "-"))
                    print(
                        "average speed: {:.3f}(images/sec)".format(
                            np.mean(self.throughoutput_list)
                        )
                    )
                    print("-".ljust(66, "-"))

        return callback
