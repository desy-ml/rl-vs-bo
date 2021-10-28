from concurrent.futures import ThreadPoolExecutor
import itertools
import logging
from pathlib import Path
import time

import pydoocs
import numpy as np
from tqdm import tqdm
import yaml

from doocsrecorder import DiskWriter, WandBWriter


directory = "/home/ttflinac/Desktop/mskrl"

scan_space = {
    "FLASH.MAGNETS/MAGNET.ML/V7SMATCH/CURRENT.SP": np.linspace(0.68, 0.79, num=3),
    "FLASH.MAGNETS/MAGNET.ML/H10SMATCH/CURRENT.SP": np.linspace(1.95, 2.32, num=3),
    "FLASH.MAGNETS/MAGNET.ML/H12SMATCH/CURRENT.SP": np.linspace(-0.0125, 0.1825, num=3),
    "FLASH.MAGNETS/MAGNET.ML/V14SMATCH/CURRENT.SP": np.linspace(0.84, 1.0, num=3),
}

limit = 0.9
show_stoppers = {
    "FLASH.DIAG/BLM/8L.TCOL/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/8R.TCOL/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/2L.ECOL/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/2R.ECOL/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/3L.ECOL/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/3R.ECOL/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/2L.ORS/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/2R.ORS/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/7L.ORS/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/7R.ORS/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/12ORS/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/1L.SFUND1/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/1R.SFUND1/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/1L.SFUND1/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/1R.SFUND1/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/1L.SFUND2/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/1R.SFUND2/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/1L.SFUND3/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/1R.SFUND3/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/1L.SFUND4/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/1R.SFUND4/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/1L.SFELC/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/1R.SFELC/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/3SFELC/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/4SFELC/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/3SDUMP/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/14L.SMATCH/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/14R.SMATCH/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/1L.UND1/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/1R.UND1/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/5L.UND1/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/5R.UND1/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/1L.UND2/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/1R.UND2/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/5L.UND2/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/5R.UND2/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/1L.UND3/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/1R.UND3/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/5L.UND3/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/5R.UND3/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/1L.UND4/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/1R.UND4/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/5L.UND4/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/5R.UND4/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/1L.UND5/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/1R.UND5/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/5L.UND5/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/5R.UND5/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/1L.UND6/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/1R.UND6/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/5L.UND6/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/5R.UND6/SIGNAL.FLASH1": limit,
    "FLASH.DIAG/BLM/2EXP/SIGNAL.FLASH1": limit,
}

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S"
)


def write_settings(settings, wait_for_busy=True):
    logging.info(f"Moving to {settings}")

    for channel, value in settings.items():
        pydoocs.write(channel, value)
    
    if not wait_for_busy:
        return

    time.sleep(1.0)

    busy_channels = ["/".join(channel.split("/")[:-1] + ["BUSY"]) for channel in settings.keys()]

    while any(pydoocs.read(channel)["data"] for channel in busy_channels):
        time.sleep(0.25)


def read_data(channels):
    executor = ThreadPoolExecutor()
    futures = {channel: executor.submit(pydoocs.read, channel) for channel in channels}

    data = {}
    for channel, future in futures.items():
        try:
            data[channel] = future.result()
        except pydoocs.DoocsException as e:
            logging.warning(f"Got exception {e} for channel {channel}")
            data[channel] = e
                
    return data


def check_showstoppers(show_stoppers):
    channels = show_stoppers.keys()    
    data = read_data(channels)
    for channel in channels:
        if data[channel]["data"] > show_stoppers[channel]:
            logging.warning(f"Show stopper {channel} violated")
            return False
    return True


def main():
    Path(directory).mkdir(parents=True, exist_ok=True)
    configpath = "sase.yml"

    logging.info(f"Starting data collection to {directory}")

    channels, values = zip(*scan_space.items())

    with open(configpath) as f:
        config = yaml.safe_load(f)
    data_channels = config["channels"]
    data_channels = sorted(filter(lambda x: x is not None , data_channels)) # TODO: Fix None entry

    writers = [
        DiskWriter(directory, pulses_per_file=100),
        # WandBWriter(name=directory.split("/")[-1])
    ]

    i = 0
    total = 1
    for dim in scan_space.values():
        total *= len(dim)
    for actuators in tqdm(itertools.product(*values), total=total):
        settings = {channel: value for channel, value in zip(channels, actuators)}

        write_settings(settings, wait_for_busy=False)
        
        # Wait for GMD to react and check show stopper violation in the meantime
        t0 = time.time()
        while time.time() - t0 < 30:
            if not check_showstoppers(show_stoppers):
                choice = input("Read (r), continue (c) with next setting or abort (a) scan?")
                if choice == "r":
                    pass
                elif choice == "c":
                    continue
                elif choice == "a":
                    exit()
                else:
                    raise ValueError(f"Choice {choice} was not an option!")

        for _ in range(10):
            logging.info(f"Reading data (pulse={i})")
            data = read_data(data_channels)
            sample = {"settings": settings, "data": data}
            
            for writer in writers:
                logging.info("Writing data")
                writer.write(i, sample)
            
            i += 1

    logging.info(f"Finished collecting samples!")


if __name__ == "__main__":
    main()
