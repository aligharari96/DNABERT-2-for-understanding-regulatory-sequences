import argparse
from utils import *

def main(args):
    if args.task == 1:
        run_task1(args)
    else:
        run_task2(args)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--task", type=int, default=1)
    parser.add_argument("--results_folder", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="data/")
    args = parser.parse_args()
    main(args)