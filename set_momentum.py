import argparse

import pydoocs


def main():
    parser = argparse.ArgumentParser("Set all momentum fields of our magnets of some beam energy.")
    parser.add_argument("momentum", type=float, help="momentum value to write to magnets")

    momentum = parser.parse_args().momentum

    for magnet in ["AREAMQZM1", "AREAMQZM2", "AREAMCVM1", "AREAMQZM3", "AREAMCHM1"]:
        channel = f"SINBAD.MAGNETS/MAGNET.ML/{magnet}/MOMENTUM.SP"
        print(f"Writing {momentum} to {channel}")
        pydoocs.write(channel, momentum)


if __name__ == "__main__":
    main()
