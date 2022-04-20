#!/bin/sh

weights=("resnet18-f37072fd.pth" "resnet34-b627a593.pth" "resnet50-0676ba61.pth" "resnet101-63fe2227.pth" "resnet152-394f9c45.pth")

for w in "${weights[@]}"; do
    FILE="`dirname "$0"`/predictor/weights/$w"
    if [ -f "$FILE" ]; then
        echo "$FILE already exists."
    else
        wget -P `dirname "$0"`/predictor/weights "https://download.pytorch.org/models/$w"
    fi
done

echo "successfully downloaded model weights."