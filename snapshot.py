import argparse
import pickle
import time

import numpy as np
import pydoocs

from config import auxiliary_channels
from environments import machine



def switch_cathode_laser(self, setto):
    """Sets the bool switch of the cathode laser event to setto and waits a second."""
    address = "SINBAD.DIAG/TIMER.CENTRAL/MASTER/EVENT5"
    bits = pydoocs.read(address)["data"]
    bits[0] = 1 if setto else 0
    pydoocs.write(address, bits)
    time.sleep(1)


def cathode_laser_on(self):
    self.switch_cathode_laser(True)


def cathode_laser_off(self):
    self.switch_cathode_laser(False)


def capture_screen(self):
    return pydoocs.read(self.screen_channel + "IMAGE_EXT_ZMQ")["data"]


def capture_interval(self, n, dt):
    images = []
    for _ in range(n):
        images.append(self.capture_screen())
        time.sleep(dt)
    return np.array(images)


def snapshot(filename, measure_beam=False):
    # Read auxiliary channels
    data = {}
    for channel in auxiliary_channels:
        print(f"Reading {channel}")
        data[channel] = pydoocs.read(channel)["data"]

    # Measure beam
    if measure_beam:
        # Laser off
        cathode_laser_off()
        background_images = capture_interval(n=10, dt=0.1)
        median_background = np.median(background_images.astype("float64"), axis=0)

        # Laser on
        cathode_laser_on()
        beam_images = capture_interval(n=10, dt=0.1)
        median_beam = np.median(beam_images.astype("float64"), axis=0)

        removed = (median_beam - median_background).clip(0, 2**16-1)
        flipped = np.flipud(removed)

        data["background_images"] = background_images
        data["beam_images"] = beam_images
        data["median_background"] = median_background
        data["median_beam"] = median_beam
        data["screen_data"] = flipped

    path = filename + ".pkl"
    print(f"Writing data to {path}")
    with open(path, "wb") as f:
        pickle.dump(data, f)


def video(filename, dt=0.5, measure_beam=False):
    try:
        i = 0
        while True:
            t1 = time.time()
            framename = filename + f"-{i:06d}"
            snapshot(framename, measure_beam=measure_beam)

            while time.time() < t1 + dt:
                pass

            i += 1
    except KeyboardInterrupt:
        pass


def main():
    parser = argparse.ArgumentParser("Save current values of interesting upstream channels.")
    parser.add_argument("filename", help="file to save data to")
    parser.add_argument("--measure_beam", type=bool, default=False, help="measure beam parameters")
    parser.add_argument("--dt", type=float, default=None, help="record snapshot every dt seconds")
    args = parser.parse_args()

    if args.dt == None:
        print("Taking snapshot")
        time.sleep(1)
        snapshot(args.filename, measure_beam=args.measure_beam)
    else:
        print("Recording video")
        time.sleep(1)
        video(args.filename, dt=args.dt, measure_beam=args.measure_beam)


if __name__ == "__main__":
    main()
