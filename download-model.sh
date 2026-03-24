#!/usr/bin/env bash
set -euo pipefail

MODEL="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${MODEL}.tar.bz2"
DIR="models"

if [ -d "${DIR}/${MODEL}" ]; then
    echo "Model already downloaded: ${DIR}/${MODEL}"
    exit 0
fi

mkdir -p "$DIR"
echo "Downloading ${MODEL} (~640 MB)..."
curl -L "$URL" -o "${DIR}/${MODEL}.tar.bz2"

echo "Extracting..."
tar xjf "${DIR}/${MODEL}.tar.bz2" -C "$DIR"
rm "${DIR}/${MODEL}.tar.bz2"

echo "Done. Model at: ${DIR}/${MODEL}"
ls -lh "${DIR}/${MODEL}/"
