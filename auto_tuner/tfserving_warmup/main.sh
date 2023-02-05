#!/bin/bash

# turn on bash's job control
set -m

# Start tfserving process and put it in the background
tensorflow_model_server --port=8500 --rest_api_port=8501 "$@" &

sleep 0.5

start=`date +%s`

flag=0
while [ "$flag" -eq 0 ]
do
    resp=$(curl 127.0.0.1:8501/v1/models/resnet | ./jq -r ".model_version_status[0].state")
    if [[ "$resp" = "AVAILABLE" ]]
    then
            flag=1
    else
            sleep 0.25
    fi
done

end=`date +%s`

runtime=$((end-start))
echo "Took $runtime seconds to load"

python warmup.py


touch /app/warmup_done

# now we bring the primary process back into the foreground and leave it there
fg %1
