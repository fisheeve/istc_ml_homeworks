import argparse
import sys


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f')
    args = parser.parse_args()
    return args


def main(args):
    for line in sys.stdin:
        x = float(line[:-1])
        print(eval(args.f))


if __name__ == '__main__':
    args = args_parse()
    main(args)
