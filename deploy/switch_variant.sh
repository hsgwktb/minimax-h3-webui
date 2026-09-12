#!/bin/bash
# Swap the resident MiniMax-H3 partition and wait until it is serving.
# Writes /content/h3/variant.json so the gateway (and the web UI) can report
# progress: {current, state: switching|ready|error, target, since}.
# usage: switch_variant.sh fl2va|ref2va
V="${1:?variant required (fl2va|ref2va)}"
D=/content/h3
LOG="$D/server_${V}.log"

st() {  # st <current> <state> <target>
  python3 -c "import json,sys,time;json.dump({'current':(sys.argv[2] or None),'state':sys.argv[3],'target':(sys.argv[4] or None),'since':time.time()},open(sys.argv[1],'w'))" \
    "$D/variant.json" "$1" "$2" "$3"
}

st "" switching "$V"
pkill -f 'sglang serve' 2>/dev/null
sleep 3
pkill -9 -f 'sglang serve' 2>/dev/null
sleep 2

: > "$LOG"
nohup bash "$D/serve_variant.sh" "$V" > "$LOG" 2>&1 &

# /health returns 503 while loading/warming up, 200 once requests are accepted
for _ in $(seq 1 240); do
  sleep 5
  CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:30010/health || true)
  if [ "$CODE" = "200" ]; then
    st "$V" ready ""
    exit 0
  fi
done

st "" error "$V"
exit 1
