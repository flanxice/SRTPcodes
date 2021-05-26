#! /bin/bash
nohup python3 ./device1_acc.py & >> ./log_acc/errorlog1.log
# pid=$!
# taskset -a -pc 0 $pid
# cpulimit -p $pid -l 100
