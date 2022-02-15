import argparse
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--file_root', default='', type = str)
    args = parser.parse_args()

    os.remove(args.file_root)
    print('Destructed! ')