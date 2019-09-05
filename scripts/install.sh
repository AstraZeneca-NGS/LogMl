#!/bin/bash

INSTALL_DIR="$HOME/logml"

# Get the srouce code directory
DIR=$( cd $(dirname "$0") ; pwd -P)
SRC_DIR=$cd "$DIR ; cd .. ; pwd -P"

cd "$WORKSPACE_DIR/.."
git clone https://github.com/AstraZeneca-NGS/LogMl.git

mkdir "$INSTALL_DIR"
cd "$INSTALL_DIR"

for d in config logml requirements.txt scripts src tests; do
  ln -s "$SRC_DIR/$d"
done

cd "$INSTALL_DIR"
virtualenv -p python3 .
. ./bin/activate
pip install -r requirements.txt
