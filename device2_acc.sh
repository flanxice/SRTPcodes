nohup python3 ./device2_acc.py & >> ./log_acc/errorlog2.log
# pid=$!
# taskset -a -pc 0 $pid
# cpulimit -p $pid -l 100
