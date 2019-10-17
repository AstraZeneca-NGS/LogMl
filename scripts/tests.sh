#!/bin/bash -eu
set -o pipefail

DIR="$(cd $(dirname $0) ; pwd -P)"

export PS1=""

source ./bin/activate

test_integration="${1:-false}"

# Only perform one test
TEST_NAME=""
# TEST_NAME="TestLogMl.test_dataset_transform_002"

echo
echo
if [ -z "$TEST_NAME" ]; then
    echo "Unit tests: All "
    time coverage run src/tests.py -v --failfast 2>&1 | tee tests.unit.out
else
    echo "Unit test: '$TEST_NAME' "
    time coverage run src/tests.py -v --failfast "$TEST_NAME" 2>&1 | tee tests.unit.out
fi

coverage report -m --fail-under=60 --omit='*lib/*' 2>&1 | tee -a tests.unit.out

# Should we do integration testing?
if [ "$test_integration" eq 'false']; do
  exit
fi

echo
echo
echo "Integration tests"

./src/tests_integration.py -v --failfast 2>&1 | tee tests.integration.out

echo
echo
echo "Done: All tests passed"
