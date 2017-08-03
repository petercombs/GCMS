import pandas as pd
from sys import argv

if __name__ == "__main__":
    out = pd.DataFrame()
    for fname in argv[1:]:
        df = pd.read_table(fname, skiprows=6, index_col=0)
        out[fname.split('.')[0]] = df.pval


