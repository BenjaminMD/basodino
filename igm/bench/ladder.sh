#!/usr/bin/env bash
# ladder.sh STAGE: the benchmark batches run on s03 on 2026-10-06 (GPU 1 unless noted).
# Each line is one batch: python bench.py MODE DX N [YEARS] [GPU]; rows land in results.csv.
# Run as: source ../env.sh && nrun bash ladder.sh singles   (then: concurrency, bothgpus)
set -u
cd "$(dirname "$0")"
case "$1" in
singles)        # one run alone on the GPU
    python bench.py offline 20 1          # (results.csv row 1 is the same batch run by hand first)
    python bench.py offline 10 1
    python bench.py online 20 1
    python bench.py online 10 1
    python bench.py reference 20 1 10
    # python bench.py reference 10 1 10   # stopped by hand after 10 min at year 2.1;
    #                                       # 10 years would take about an hour
    ;;
concurrency)    # N identical runs started together on GPU 1
    for dx in 20 10; do
        for mode in online offline; do
            for n in 4 8 16; do python bench.py $mode $dx $n; done
        done
    done
    ;;
bothgpus)       # the same batch on GPU 0 and GPU 1 at the same time
    python bench.py "$2" "$3" "$4" 100 0 &
    python bench.py "$2" "$3" "$4" 100 1
    wait
    ;;
esac
