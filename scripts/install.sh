#!/bin/bash -eu
set -o pipefail

INSTALL_DIR="$HOME/logml"

# Get the srouce code directory
DIR=$( cd $(dirname "$0") ; pwd -P)
echo "DIR=$DIR"
SRC_DIR=$(cd "$DIR/.." ; pwd -P)

# Create install dir
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

for d in config logml requirements.txt scripts src tests; do
  ln -s "$SRC_DIR/$d" || true
done

cd "$INSTALL_DIR"
export PS1=""
virtualenv -p python3 .
. ./bin/activate
pip install -r requirements.txt
