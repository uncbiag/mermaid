#!/usr/bin/env bash
set -e

PYCMD=${PYCMD:="python"}

pushd "$(dirname "$0")"

echo "Running pyreg tests for: finite differences"
$PYCMD test_finite_differences.py $@

echo "Running pyreg tests for: module_parameters"
$PYCMD test_module_parameters.py $@

echo "Running pyreg tests for: stn"
$PYCMD test_stn.py $@

popd
