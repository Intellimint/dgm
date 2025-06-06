#!/bin/bash

# Directory to store logs and diffs
RESULTS_DIR="overnight_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Set the end time (8 hours from now)
END_TIME=$(( $(date +%s) + 8*60*60 ))

# Main loop
RUN=1
while [ $(date +%s) -lt $END_TIME ]; do
    RUN_DIR="$RESULTS_DIR/run_$(printf '%03d' $RUN)_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$RUN_DIR"
    
    echo "[INFO] Starting self-improvement run $RUN at $(date)" | tee -a "$RESULTS_DIR/summary.log"
    
    # Save a copy of coding_agent.py before the run
    cp coding_agent.py "$RUN_DIR/coding_agent_before.py"
    
    # Run the self-improvement step with test_problem entry point
    python self_improve_step.py --entry test_problem --polyglot --force_rebuild > "$RUN_DIR/self_improve.log" 2>&1
    
    # Save a copy of coding_agent.py after the run
    cp coding_agent.py "$RUN_DIR/coding_agent_after.py"
    
    # Show and save the diff
    diff "$RUN_DIR/coding_agent_before.py" "$RUN_DIR/coding_agent_after.py" > "$RUN_DIR/coding_agent.diff"
    
    # Debug: Show file sizes and first few lines
    echo "[DEBUG] Before file size: $(wc -l < "$RUN_DIR/coding_agent_before.py")" | tee -a "$RESULTS_DIR/summary.log"
    echo "[DEBUG] After file size: $(wc -l < "$RUN_DIR/coding_agent_after.py")" | tee -a "$RESULTS_DIR/summary.log"
    echo "[DEBUG] First few lines of diff:" | tee -a "$RESULTS_DIR/summary.log"
    head -n 5 "$RUN_DIR/coding_agent.diff" | tee -a "$RESULTS_DIR/summary.log"
    
    # Check if there are any real changes (not just test changes)
    if grep -q "TEST CHANGE" "$RUN_DIR/coding_agent.diff"; then
        echo "[WARNING] Only test changes detected in run $RUN" | tee -a "$RESULTS_DIR/summary.log"
    fi
    
    # Check if test results exist and show them
    if [ -f "$RUN_DIR/test_results.txt" ]; then
        echo "[INFO] Test results from run $RUN:" | tee -a "$RESULTS_DIR/summary.log"
        cat "$RUN_DIR/test_results.txt" | tee -a "$RESULTS_DIR/summary.log"
    else
        echo "[WARNING] No test results found for run $RUN" | tee -a "$RESULTS_DIR/summary.log"
    fi
    
    echo "[INFO] Finished run $RUN at $(date). Diff summary:" | tee -a "$RESULTS_DIR/summary.log"
    diffstat "$RUN_DIR/coding_agent.diff" | tee -a "$RESULTS_DIR/summary.log"
    echo "----------------------------------------" | tee -a "$RESULTS_DIR/summary.log"
    
    RUN=$((RUN+1))
    sleep 10  # Wait 10 seconds between runs

done

echo "[INFO] Continuous self-improvement finished at $(date)" | tee -a "$RESULTS_DIR/summary.log" 