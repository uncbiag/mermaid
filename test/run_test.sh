#!/usr/bin/env bash
set -e

PYCMD=${PYCMD:="python"}

pushd "$(dirname "$0")"

echo "Running pyreg tests for: finite differences"
$PYCMD test_finite_differences.py $@

popd
