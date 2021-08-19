from datetime import datetime
import pickle
import time

import numpy as np
import pydoocs
from scipy.ndimage import minimum_filter1d, uniform_filter1d


def read_screen():
    """Get pixel data from the screen."""

    channel = "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/IMAGE_EXT_ZMQ"

    switch_cathode_laser(False)
    backgrounds = capture(10, channel)
    background = backgrounds.mean(axis=0)
    
    switch_cathode_laser(True)
    beams = capture(10, channel)
    beam = beams.mean(axis=0)

    image = beam - background
    
    return image


def capture(n, channel):
    images = []
    for _ in range(n):
        response = pydoocs.read(channel)
        flippedud = np.flipud(response["data"])
        flippedlr = np.fliplr(flippedud)
        images.append(flippedlr)
        time.sleep(0.1)
    return np.array(images)


def switch_cathode_laser(setto):
    """Sets the bool switch of the cathode laser event to setto and waits a second."""

    address = "SINBAD.DIAG/TIMER.CENTRAL/MASTER/EVENT5"
    bits = pydoocs.read(address)["data"]
    bits[0] = 1 if setto else 0
    pydoocs.write(address, bits)
    time.sleep(1)


def beam_parameters(screen_data, pixel_size, binning):
    parameters = np.empty(4)
    for axis in [0, 1]:
        profile = screen_data.sum(axis=axis)
        minfiltered = minimum_filter1d(profile, size=5, mode="nearest")
        filtered = uniform_filter1d(minfiltered, size=5, mode="nearest")

        half_values, = np.where(filtered >= 0.5 * filtered.max())

        if len(half_values) > 0:
            fwhm_pixel = half_values[-1] - half_values[0]
            center_pixel = half_values[0] + fwhm_pixel / 2
        else:
            fwhm_pixel = 42     # TODO: Figure out what to do with these
            center_pixel = 42

        parameters[axis] = (center_pixel - len(filtered) / 2) * pixel_size[axis] * binning
        parameters[axis+2] = fwhm_pixel / 2.355 * pixel_size[axis] * binning
        
    parameters[1] = -parameters[1]

    return parameters


if __name__ == "__main__":
    n = 1
    log = []

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    for i in range(n):
        screen_resolution = (2448, 2040)
        pixel_size = (3.3198e-6, 2.4469e-6)
        binning = 4

        screen_data = read_screen()
        
        parameters = beam_parameters(
            screen_data,
            pixel_size=pixel_size,
            binning=binning
        )

        log.append((parameters, screen_data))
        print(i, parameters)

        time.sleep(20.0)
    
    with open(f"measurements/beamparameters-{timestamp}.pkl", "wb") as f:
        pickle.dump(log, f)
