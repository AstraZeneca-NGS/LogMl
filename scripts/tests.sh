#!/bin/bash -eu
set -o pipefail

DIR="$(cd $(dirname $0) ; pwd -P)"

export PS1=""

source ./bin/activate

TEST_NAME=""
# TEST_NAME="TestLogMl.test_dataset_006"

if [ -z "$TEST_NAME" ]; then
    time coverage run src/tests.py -v --failfast 2>&1 | tee tests.out
else
    time coverage run src/tests.py -v --failfast "$TEST_NAME" 2>&1 | tee tests.out
fi

coverage report -m --fail-under=60 --omit='*lib/*' 2>&1 | tee -a tests.out
echo "Done: All tests passed"
