import argparse

import pydoocs


def main():
    subchannels = [
        "AREAMQZM1/STRENGTH.SP",
        "AREAMQZM2/STRENGTH.SP",
        "AREAMCVM1/KICK_MRAD.SP",
        "AREAMQZM3/STRENGTH.SP",
        "AREAMCHM1/KICK_MRAD.SP"
    ]

    parser = argparse.ArgumentParser("Set all momentum fields of our magnets of some beam energy.")
    for subchannel in subchannels:
        magnet = subchannel.split("/")[0]
        parser.add_argument(magnet, type=float, help=f"value to set {subchannel} to")

    args = vars(parser.parse_args())
    
    for subchannel in subchannels:
        magnet = subchannel.split("/")[0]
        value = args[magnet]
        channel = f"SINBAD.MAGNETS/MAGNET.ML/{subchannel}"
        print(f"Writing {value} to {channel}")
        pydoocs.write(channel, value)


if __name__ == "__main__":
    main()
