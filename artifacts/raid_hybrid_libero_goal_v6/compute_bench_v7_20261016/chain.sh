#!/bin/zsh
# Start the v7 compute benchmark only after the pilot finisher confirmed closure and an empty Runpod account.
cd "$(dirname "$0")"
export PATH=/usr/local/bin:$PATH
OPS=/Users/ConstiX/.config/raid-hybrid-v6/ops-venv/bin/python
PILOT=../vpp2_pilot_v5_20261014
until [ -f $PILOT/final_status.json ] && ! pgrep -f "vpp2_pilot_v5_20261014/finisher.sh" >/dev/null; do sleep 120; done
python3 -c "
import json,sys; s=json.load(open('$PILOT/final_status.json'))
sys.exit(0 if s.get('service_closed') and not s.get('runpod_pods') and not s.get('runpod_volumes') else 5)" || { echo "pilot not cleanly closed; benchmark not started"; exit 5; }
nohup $OPS execute.py >> execute.log 2>&1 &
PID=$!; echo $PID > driver.pid
nohup caffeinate -i -w $PID >/dev/null 2>&1 &
echo "BENCH_LAUNCHED $(date -u +%H:%M) pid $PID"
while kill -0 $PID 2>/dev/null; do sleep 120; done
grep -E "^(GPU|ACCEPTANCE_PASSED|BENCH_WORKER_STARTED|BENCH_VERIFIED_AND_CLEANED|STOPPED)" execute.log | tail -6
