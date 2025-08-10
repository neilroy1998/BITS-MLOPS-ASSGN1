#!/usr/bin/env bash
set -euo pipefail

CNT="${1:-housing-aio}"                 # container name (arg 1 optional)
API="http://localhost:8000"
PROM="http://localhost:9090"
GRAF="http://localhost:3000"
MLF="http://localhost:5002"

say() { printf "\n🔹 %s\n" "$*"; }

http() {  # method url [data]
  local m="$1" u="$2" d="${3-}"
  if [ -n "$d" ]; then
    curl -s -X "$m" -H 'Content-Type: application/json' -d "$d" \
      -w ' | code=%{http_code} time=%{time_total}s\n' "$u"
  else
    curl -s -X "$m" "$u" -w ' | code=%{http_code} time=%{time_total}s\n'
  fi
}

randf() { # rand float in [min,max]
  awk -v min="$1" -v max="$2" -v r="$RANDOM" 'BEGIN{printf "%.4f", min+(max-min)*(r/32767)}'
}

# ---------- 0) quick host reachability ----------
say "Host reachability"
echo "API /health        : $(curl -s -o /dev/null -w '%{http_code}' $API/health)"
echo "Prometheus /up     : $(curl -s -o /dev/null -w '%{http_code}' $PROM/api/v1/query?query=up)"
echo "Grafana /api/health: $(curl -s -o /dev/null -w '%{http_code}' $GRAF/api/health)"
echo "MLflow /           : $(curl -s -o /dev/null -w '%{http_code}' $MLF/)"

# ---------- 1) generate traffic ----------
say "10x GET /health"
for i in {1..10}; do
  ts=$(date '+%H:%M:%S'); echo "[$ts] GET /health  #$i"
  http GET "$API/health"; sleep 0.2
done

say "20x POST /predict (randomized payloads)"
for i in {1..20}; do
  MedInc=$(randf 4.0 12.0)
  HouseAge=$(randf 1 50)
  AveRooms=$(randf 3 9)
  AveBedrms=$(randf 0.5 2.5)
  Population=$(randf 100 5000)
  AveOccup=$(randf 1 5)
  Latitude=$(randf 32 42)
  Longitude=$(randf -124 -114)

  payload=$(jq -n \
    --argjson MedInc "$MedInc" --argjson HouseAge "$HouseAge" \
    --argjson AveRooms "$AveRooms" --argjson AveBedrms "$AveBedrms" \
    --argjson Population "$Population" --argjson AveOccup "$AveOccup" \
    --argjson Latitude "$Latitude" --argjson Longitude "$Longitude" \
    '{MedInc:$MedInc,HouseAge:$HouseAge,AveRooms:$AveRooms,AveBedrms:$AveBedrms,Population:$Population,AveOccup:$AveOccup,Latitude:$Latitude,Longitude:$Longitude}'
  )

  ts=$(date '+%H:%M:%S'); echo "[$ts] POST /predict  #$i"
  http POST "$API/predict" "$payload"; sleep 0.2
done

# ---------- 2) quick summaries ----------
say "Summaries from /metrics (API)"
curl -s "$API/metrics" | egrep '^http_requests_total|^predicted_house_value_(count|sum)$|^predicted_house_value_bucket{le=' | head -n 30

say "Prometheus spot-checks"
echo "RPS by handler:"; curl -s "$PROM/api/v1/query?query=sum%20by%20(handler)(rate(http_requests_total%5B1m%5D))" | jq -r '.data.result[]|[.metric.handler, .value[1]]|@tsv'
echo "P95 latency:";   curl -s "$PROM/api/v1/query?query=histogram_quantile(0.95,%20sum%20by%20(le)(rate(http_request_duration_seconds_bucket%5B5m%5D)))" | jq -r '.data.result[].value[1]'
echo "Pred value mean (5m):"; curl -s "$PROM/api/v1/query?query=sum(increase(predicted_house_value_sum%5B5m%5D))%2Fsum(increase(predicted_house_value_count%5B5m%5D))" | jq -r '.data.result[].value[1]'

# ---------- 3) inside-container checks ----------
say "Inside-container checks (docker exec $CNT)"
docker exec "$CNT" sh -lc '
set -e
printf "API in-container /metrics status: %s\n" "$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/metrics)"
printf "Prometheus up(): %s\n" "$(curl -s http://127.0.0.1:9090/api/v1/query?query=up | grep -o "\"status\":\"success\"" || true)"
printf "Grafana health: %s\n" "$(curl -s http://127.0.0.1:3000/api/health | tr -d "\n")"
printf "MLflow root: %s\n" "$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:5002/)"
'

say "Done ✅"
