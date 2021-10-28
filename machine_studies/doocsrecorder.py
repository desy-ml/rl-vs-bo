import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import logging
import os
import pathlib
import pickle
import time

import numpy as np
import pydoocs
import wandb
import yaml


# Setup logging (to console)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class DoocsRecorder:

    def __init__(self, channels, dt, writers=[]):
        self.channels = channels
        self.dt = dt
        self.writers = writers

        self._executor = ThreadPoolExecutor()

    def run(self):
        for pulse, data in enumerate(self.read()):
            self.write(pulse, data)
    
    def read(self):
        t1 = time.time()
        while True:
            futures = {channel: self._executor.submit(pydoocs.read, channel) for channel in self.channels}

            data = {}
            for channel, future in futures.items():
                try:
                    data[channel] = future.result()
                except pydoocs.DoocsException as e:
                    logger.warning(f"Got exception {e} for channel {channel}")
                    data[channel] = e
                        
            yield data

            t2 = time.time()
            dt_left = self.dt - (t2 - t1)
            if dt_left > 0:
                time.sleep(dt_left)
                t1 += self.dt
            else:
                logger.warning(f"Exceeded intended dt of {self.dt:.2f}: {t2-t1:.2f}")
                t1 = t2
    
    def write(self, pulse, data):
        for writer in self.writers:
            writer.write(pulse, data)


class Writer:
    
    def write(self, pulse, data):
        raise NotImplementedError


class PrintWriter(Writer):

    def write(self, pulse, data):
        print(f"At pulse {pulse}")
        for channel, response in data.items():
            if isinstance(response, dict):
                print(f"  {channel} -> {response['data']}")
            else:
                print(f"  {channel} -> {response}")


class DiskWriter(Writer):
    
    def __init__(self, directory, pulses_per_file=1000):
        logger.info(f"Setting up Disk Writer to {directory} every {pulses_per_file} pulses")

        self.directory = directory
        self.pulses_per_file = pulses_per_file

        self._data = {}

        self._executor = ThreadPoolExecutor()

        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    
    def __del__(self):
        self._executor.shutdown(wait=True)
        if len(self._data) > 0:
            self._dump(self._data)

    def write(self, pulse, data):
        self._data[pulse] = data

        if len(self._data) == self.pulses_per_file:
            dumpable = self._data
            self._executor.submit(self._dump, dumpable)
            self._data = {}
        elif len(self._data) > self.pulses_per_file:
            raise Exception("_data has grown larger than is permitted.")
    
    def _dump(self, dumpable):
        filename = datetime.now().strftime("%Y%m%d%H%M%S") + ".pkl"
        path = os.path.join(self.directory, filename)
        with open(path, "wb") as f:
            pickle.dump(dumpable, f)
        logger.info(f"Saved file to {path}")


class WandBWriter:

    def __init__(self, name=None):
        wandb.init(project="many-sase-chlorians", entity="msk-ipc", name=name)
        logger.info(f"Setting up W&B Writer to run {wandb.run.name}")
        self._executor = ThreadPoolExecutor()
    
    def __del__(self):
        self._executor.shutdown(wait=True)

    def write(self, pulse, data):
        self._executor.submit(self._upload, pulse, data)
    
    def _upload(self, pulse, data):
        values = {}
        for channel, response in data.items():
            values[channel] = response["data"] if isinstance(response, dict) else np.nan
        
        wandb.log(values, step=pulse)


def main():
    parser = argparse.ArgumentParser("Log values from pydoocs channels.")
    parser.add_argument("configpath", help="YAML file with configuration for the run")
    args = parser.parse_args()

    with open(args.configpath) as f:
        config = yaml.safe_load(f)
    channels, dt = config["channels"], config["dt"], 

    channels = sorted(filter(lambda x: x is not None , channels)) # TODO: Fix None entry

    logger.info(f"Record {len(channels)} channels")
    logger.info(f"Read time {dt} s ({1/dt} Hz)")

    writers = []
    if "wandb" in config:
        wandb_writer = WandBWriter(name=config["wandb"]["name"])
        writers.append(wandb_writer)
    if "disk" in config:
        disk_writer = DiskWriter(
            directory=config["disk"]["directory"],
            pulses_per_file=config["disk"]["pulses_per_file"]
        )
        writers.append(disk_writer)
        
    dlogger = DoocsRecorder(channels, dt, writers=writers)

    try:
        dlogger.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
