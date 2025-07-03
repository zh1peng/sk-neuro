import argparse
import os
import time
import numpy as np

from .io import (
    run_cv_on_csv,
    run_cv_on_directory,
    run_repeated_cv_on_csv,
    save_cv_diagnostic_plot,
)
from .cv_utils import pearson_corr


def main():
    parser = argparse.ArgumentParser(description="ElasticNet CV utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_once = subparsers.add_parser("once", help="Run cross-validated prediction on a CSV file")
    p_once.add_argument("csv", help="Path to CSV file")
    p_once.add_argument("--family", default="gaussian", help="glmnet family: gaussian, binomial, multinomial")

    p_batch = subparsers.add_parser("batch", help="Run cross-validated prediction on all CSV files in a directory")
    p_batch.add_argument("directory", help="Directory containing CSV files")
    p_batch.add_argument("--family", default="gaussian", help="glmnet family")

    p_repeat = subparsers.add_parser("repeat", help="Repeated cross-validation on a CSV file")
    p_repeat.add_argument("csv", help="Path to CSV file")
    p_repeat.add_argument("repeats", type=int, help="Number of repetitions")
    p_repeat.add_argument("--family", default="gaussian", help="glmnet family")

    p_perm = subparsers.add_parser("permutation", help="Permutation testing via repeated CV")
    p_perm.add_argument("csv", help="Path to CSV file")
    p_perm.add_argument("repeats", type=int, help="Number of repetitions")
    p_perm.add_argument("--family", default="gaussian", help="glmnet family")

    p_plot = subparsers.add_parser("plot", help="Generate diagnostic plot for a CSV file")
    p_plot.add_argument("csv", help="Path to CSV file")
    p_plot.add_argument("--family", default="gaussian", help="glmnet family")

    args = parser.parse_args()

    if args.command == "once":
        start = time.time()
        r, p, e_time = run_cv_on_csv(args.csv, family=args.family)
        print(f"r: {r}, p: {p}, time: {e_time}")
        print_time(start)
    elif args.command == "batch":
        start = time.time()
        results = run_cv_on_directory(args.directory, family=args.family)
        results.to_csv("batch_test_result.csv")
        print(results)
        print_time(start)
    elif args.command == "repeat":
        start = time.time()
        res = run_repeated_cv_on_csv(args.csv, n_repeats=args.repeats, shuffle_y=False, family=args.family)
        np.save(os.path.splitext(args.csv)[0] + f"_repeat_{args.repeats}_cv_results.npy", res)
        if args.family == "gaussian":
            r, p = pearson_corr(res["mean_y_pred"], res["true_y"])
            print(f"Mean predict y across {args.repeats} repeats, correlation is: {r}, {p}")
        else:
            print(f"Mean accuracy across {args.repeats} repeats: {res['accuracy']}")
        print_time(start)
    elif args.command == "permutation":
        start = time.time()
        res = run_repeated_cv_on_csv(args.csv, n_repeats=args.repeats, shuffle_y=True, family=args.family)
        np.save(os.path.splitext(args.csv)[0] + f"_permutation_{args.repeats}_results.npy", res)
        if args.family == "gaussian":
            r, p = pearson_corr(res["mean_y_pred"], res["true_y"])
            print(f"Mean predict y across {args.repeats} repeats, correlation is: {r}, {p}")
        else:
            print(f"Mean accuracy across {args.repeats} repeats: {res['accuracy']}")
        print_time(start)
    elif args.command == "plot":
        save_cv_diagnostic_plot(args.csv, family=args.family)
    else:
        parser.print_help()


def print_time(start):
    e = int(time.time() - start)
    print("Time elapsed:{:02d}:{:02d}:{:02d}".format(e // 3600, (e % 3600 // 60), e % 60))


if __name__ == "__main__":
    main()
