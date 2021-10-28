import argparse
import itertools
import logging
import pickle
import time

import pydoocs
import numpy as np
from tqdm import tqdm


scan_space = {
    "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM1/CURRENT.SP": np.linspace(0.68, 0.79, num=3),
}

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S"
)

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


def write_settings(settings, wait_for_busy=True):
    logging.info(f"Moving to {settings}")

    for channel, value in settings.items():
        pydoocs.write(channel, value)
    
    if not wait_for_busy:
        return

    time.sleep(1.0)

    busy_channels = ["/".join(channel.split("/")[:-1] + ["BUSY"]) for channel in settings.keys()]
    ps_channels = ["/".join(channel.split("/")[:-1] + ["PS_ON"]) for channel in settings.keys()]
    
    time.sleep(3.0)
    while any(pydoocs.read(busy)["data"] or not pydoocs.read(ps)["data"] for busy, ps in zip(busy_channels, ps_channels)):
        time.sleep(0.25)


def main():
    parser = argparse.ArgumentParser("Do an Oliscan.")
    parser.add_argument("filename", help="Name of the file to write to")
    args = parser.parse_args()

    logging.info(f"Starting data collection")

    channels, values = zip(*scan_space.items())

    data = []
    total = 1
    for dim in scan_space.values():
        total *= len(dim)
    for actuators in tqdm(itertools.product(*values), total=total):
        settings = {channel: value for channel, value in zip(channels, actuators)}

        write_settings(settings, wait_for_busy=True)

        sample = read_screen(return_intermediate=True)
        data.append(sample)

    logging.info(f"Saving data")

    with open(args.filename, "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()
