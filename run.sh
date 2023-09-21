#!/bin/bash

# Check if Poetry is installed
if ! command -v poetry &> /dev/null
then
    echo "Poetry is not installed. Please install it to run this script."
    exit 1
fi

# Install dependencies
poetry install

# Define a function to run the tests
task_test() {
    poetry run pytest
}

# Define a function to run the main script
task_classify() {
    poetry run python main.py
}

usage() {
  echo "USAGE"
  echo "classify | test"
  echo ""
  exit 1
}

cmd=$1
shift || true
case "$cmd" in
classify) task_classify ;;
test) task_test ;;
*) usage ;;
esac