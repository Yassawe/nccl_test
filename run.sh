echo "l20 sector queries"
nvprof --devices 0 --profile-child-processes --print-gpu-trace --normalized-time-unit ms --profile-from-start off --concurrent-kernels on --events l2_subp0_total_read_sector_queries --csv --log-file ./csv/l20_%p.csv mpirun -np 4 --allow-run-as-root build/examples/example-2 524288

echo "l20 sector misses"
nvprof --devices 0 --profile-child-processes --print-gpu-trace --normalized-time-unit ms --profile-from-start off --concurrent-kernels on --events l2_subp0_read_sector_misses --csv --log-file ./csv/m20_%p.csv mpirun -np 4 --allow-run-as-root build/examples/example-2 524288



