#!/bin/bash

echo "=========================================="
echo "SPASM CUDA SpMV Test Script"
echo "=========================================="
echo ""

if [ $# -lt 1 ]; then
    echo "Usage: $0 <spasm_file> [iterations]"
    echo ""
    echo "Example:"
    echo "  $0 matrix.spasm"
    echo "  $0 matrix.spasm 1000"
    echo ""
    echo "To create a SPASM file from MTX format:"
    echo "  cd ../spasm_converter"
    echo "  ./mtx2spasm input.mtx output.spasm"
    exit 1
fi

SPASM_FILE=$1
ITERATIONS=${2:-100}

if [ ! -f "$SPASM_FILE" ]; then
    echo "Error: File '$SPASM_FILE' not found"
    echo ""
    echo "Please provide a valid SPASM file."
    echo "You can create one using the mtx2spasm converter:"
    echo "  cd ../spasm_converter"
    echo "  ./mtx2spasm input.mtx output.spasm"
    exit 1
fi

echo "Testing with:"
echo "  File: $SPASM_FILE"
echo "  Iterations: $ITERATIONS"
echo ""

./spasm_spmv_benchmark "$SPASM_FILE" -n "$ITERATIONS"

exit $?
