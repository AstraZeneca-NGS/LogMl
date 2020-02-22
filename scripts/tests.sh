#!/bin/bash -eu
set -o pipefail

# Setting the first command line option to anything except 'false' will run integration tests
test_integration="${1:-false}"
echo "test_integration:'$test_integration'"

# Activate virtual environment
DIR="$(cd $(dirname $0) ; pwd -P)"
export PS1=""
source ./bin/activate

# If these variables are set, only perform one unit/integration test
TEST_UNIT_NAME=""
# TEST_UNIT_NAME="-"  # Disable Unit testing
# TEST_UNIT_NAME="TestLogMl.test_dataset_preprocess_013"
TEST_INTEGRATION_NAME=""
# TEST_INTEGRATION_NAME="TestLogMlIntegration.test_class3"

#---
# Unit testing
#---
echo
echo
if [ "$TEST_UNIT_NAME" == "-" ]; then
  echo "Unit test cases disabled, skipping. "
else
  if [ -z "$TEST_UNIT_NAME" ]; then
      echo "Unit tests: All "
      time coverage run src/tests_unit.py -v --failfast 2>&1 | tee tests.unit.out
  else
      echo "Unit test: '$TEST_UNIT_NAME' "
      export TEST_UNIT_DEBUG="True"
      time coverage run src/tests_unit.py -v --failfast "$TEST_UNIT_NAME" 2>&1 | tee tests.unit.out
  fi

  coverage report -m --fail-under=60 --omit='*lib/*' 2>&1 | tee -a tests.unit.out

  echo "Test cases (unit): OK"
fi

# Should we do integration testing?
if [ "$test_integration" == 'false' ]; then
  exit
fi

#---
# Integration testing
#---
echo
echo

if [ -z "$TEST_INTEGRATION_NAME" ]; then
    echo "Integration tests: All "
    time coverage run src/tests_integration.py -v --failfast 2>&1 | tee tests.integration.out
else
    echo "Integration test: '$TEST_INTEGRATION_NAME' "
    export TEST_INTEGRATION_DEBUG="True"
    time coverage run src/tests_integration.py -v --failfast "$TEST_INTEGRATION_NAME" 2>&1 | tee tests.integration.out
fi

echo
echo
echo "Test cases (integration): OK"
