import argparse
from datetime import datetime
import logging
import os
from pathlib import Path
import pickle
import time

import dummypydoocs as pydoocs
from gym import spaces
import numpy as np


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S"
)

general_directory = "./surrogate_data"

actuator_space = spaces.Box(
    low=np.array([-30, -30, -30, -3e-3, -6e-3], dtype=np.float32),
    high=np.array([30, 30, 30, 3e-3, 6e-3], dtype=np.float32)
)

actuator_channels = [
    "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM1/",
    "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM2/",
    "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM3/",
    "SINBAD.MAGNETS/MAGNET.ML/AREAMCVM1/",
    "SINBAD.MAGNETS/MAGNET.ML/AREAMCHM1/"
]


def write_magnets(values):
    logging.info(f"Setting magnets to {values}")

    for channel, value in zip(actuator_channels[:3], values[:3]):
        pydoocs.write(channel + "STRENGTH.SP", value)
    for channel, value in zip(actuator_channels[3:], values[3:]):
        pydoocs.write(channel + "KICK_MRAD.SP", value * 1000)
    
    time.sleep(3.0)

    while any(pydoocs.read(channel + "BUSY")["data"] for channel in actuator_channels):
        time.sleep(0.25)


def read_magnets():
    readbacks = []
    for channel in actuator_channels[:3]:
        readbacks.append(pydoocs.read(channel + "STRENGTH.RBV")["data"])
    for channel in actuator_channels[3:]:
        readbacks.append(pydoocs.read(channel + "KICK_MRAD.RBV")["data"] / 1000)

    readbacks = np.array(readbacks)
    logging.info(f"Magnet readbacks are {readbacks}")

    return readbacks


def read_screen(return_intermediate=False):
    """Get pixel data from the screen."""

    channel = "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/IMAGE_EXT_ZMQ"

    switch_cathode_laser(False)
    backgrounds = capture(10, channel)
    background = np.median(backgrounds, axis=0)
    
    switch_cathode_laser(True)
    beams = capture(10, channel)
    beam = np.median(beams, axis=0)

    clean = beam - background
    image = clean.clip(0, 2**16-1)

    binning = (
        pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/BINNINGHORIZONTAL")["data"],
        pydoocs.read("SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/BINNINGVERTICAL")["data"]
    )
    
    if return_intermediate:
        return image, backgrounds, background, beams, beam, binning
    else:
        return image


def capture(n, channel):
    images = []
    for _ in range(n):
        images.append(pydoocs.read(channel)["data"].astype("float64"))
        time.sleep(0.1)
    return np.array(images)


def switch_cathode_laser(setto):
    """Sets the bool switch of the cathode laser event to setto and waits a second."""
    address = "SINBAD.DIAG/TIMER.CENTRAL/MASTER/EVENT5"
    bits = pydoocs.read(address)["data"]
    bits[0] = 1 if setto else 0
    pydoocs.write(address, bits)
    time.sleep(1)


def write_sample(directory, **kwargs):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = os.path.join(directory, f"{timestamp}.pkl")

    logging.info(f"Writing \"{filename}\"")
    logging.debug(f"Keys in file are {kwargs.keys()}")

    with open(filename, "wb") as f:
        pickle.dump(kwargs, f)


def parse_config_args():
    parser = argparse.ArgumentParser(description="Scan experimental area screen over actuators.")
    parser.add_argument("experiment", type=str, help="name of the experiment (directory)")
    parser.add_argument("n", type=int, help="number of samples to collect")

    args = parser.parse_args()

    return args.experiment, args.n


def main():
    experiment, n = parse_config_args()

    directory = os.path.join(general_directory, experiment)
    Path(directory).mkdir(parents=True, exist_ok=True)

    logging.info(f"Starting data collection for {n} samples to {directory}")

    for i in range(n):
        logging.info(f"Collecting sample {i}")

        actuators = actuator_space.sample()

        write_magnets(actuators)
        readbacks = read_magnets()

        image, backgrounds, background, beams, beam, binning = read_screen(return_intermediate=True)

        write_sample(
            directory,
            actuators=actuators,
            readbacks=readbacks,
            image=image,
            backgrounds=backgrounds,
            background=background,
            beams=beams,
            beam=beam,
            binning=binning
        )
    
    logging.info(f"Finished collecting {n} samples!")


if __name__ == "__main__":
    main()
