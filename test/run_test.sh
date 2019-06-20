#!/usr/bin/env bash
set -e

PYCMD=${PYCMD:="python"}

pushd "$(dirname "$0")"

echo "Running mermaid tests for: finite differences"
$PYCMD test_finite_differences.py $@

echo "Running mermaid tests for: module_parameters"
$PYCMD test_module_parameters.py $@

echo "Running mermaid tests for: stn"
$PYCMD test_stn_cpu.py $@
$PYCMD test_stn_gpu.py $@

echo "Running mermaid tests for registrations"
$PYCMD test_registration_algorithms.py $@

popd
