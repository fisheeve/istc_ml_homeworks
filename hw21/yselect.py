import pandas as pd
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("csv_file", type=str)
parser.add_argument('-c', type=int)

args = parser.parse_args()

data = pd.read_csv(args.csv_file)
data = data[data.columns[args.c]].values
data = np.array(data, dtype=str)

print("\n".join(data))
