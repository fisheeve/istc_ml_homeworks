import argparse
import sys
import pandas as pd


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-new', type=str)
    parser.add_argument("csv_file", type=str)
    parser.add_argument('-name', type=str, required=True)
    args = parser.parse_args()
    return args


def main(args):
    values = []
    data = pd.read_csv(args.csv_file)

    if args.new == None
        for line in sys.stdin:
            values.append(line[:-1])
    else:
        for line in args.new:
            values.append(line[:-1])

    data[args.name] = values
    print(data)
    data.to_csv("./new")


if __name__ == '__main__':
    args = args_parse()
