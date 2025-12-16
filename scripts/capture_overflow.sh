#!/bin/bash
# Script to run property tests repeatedly and capture the expression that causes stack overflow

LOG_FILE="/tmp/proptest_capture.log"
EXPR_FILE="/tmp/last_tested_expr.txt"

echo "Running property tests until stack overflow..."
echo "Expressions are logged to: $EXPR_FILE"
echo "Run log at: $LOG_FILE"
echo ""

rm -f "$LOG_FILE"
rm -f "$EXPR_FILE"

for i in $(seq 1 100); do
    echo "Run $i/100..." 
    
    # Run the test - it will write expressions to EXPR_FILE before processing
    result=$(cargo test -p cas_engine --test property_tests 2>&1)
    exit_code=$?
    
    if echo "$result" | grep -q "overflow"; then
        echo ""
        echo "========================================"
        echo "STACK OVERFLOW DETECTED on run $i!"
        echo "========================================"
        echo ""
        echo "Last expression before overflow:"
        cat "$EXPR_FILE"
        echo ""
        echo ""
        echo "Saving to $LOG_FILE"
        echo "Run $i - Stack Overflow" >> "$LOG_FILE"
        echo "Expression:" >> "$LOG_FILE"
        cat "$EXPR_FILE" >> "$LOG_FILE"
        echo "" >> "$LOG_FILE"
        echo "========================================"
        exit 1
    fi
    
    if [ $exit_code -ne 0 ]; then
        echo "Test failed (non-overflow) on run $i"
        echo "$result" | tail -20
        exit 1
    fi
done

echo ""
echo "All 100 runs completed without stack overflow!"
echo "The tests are stable."
