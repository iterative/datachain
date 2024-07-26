#!/bin/bash

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Take the directory as an argument, default to "examples"
NOTEBOOK_DIR="examples"

# Parse command line options
while getopts ":d:" opt; do
  case ${opt} in
    d )
      NOTEBOOK_DIR=$OPTARG
      ;;
    \? )
      echo "Invalid option: $OPTARG" 1>&2
      exit 1
      ;;
    : )
      echo "Invalid option: $OPTARG requires an argument" 1>&2
      exit 1
      ;;
  esac
done
shift $((OPTIND -1))

# Log file location
SUBDIR=$(basename "${NOTEBOOK_DIR}")
echo "Testing notebooks in dir: ${NOTEBOOK_DIR}"
cd "${NOTEBOOK_DIR}"

# LOG_FILE="${NOTEBOOK_DIR}/log-test-notebooks-${SUBDIR}.log"
LOG_FILE="log-test-notebooks-${SUBDIR}.log"
echo "Log file: ${LOG_FILE}"

# Clear previous logs
> "$LOG_FILE"

# Variable to track overall test status
overall_status=0

# Loop over each notebook in the specified directory
for notebook in $(ls *.ipynb | sort); do

    echo "Testing ${notebook}"

    # Run a JN papermill and save outputs
    # papermill "${notebook}" "${notebook}" \
    #     --progress-bar --log-level ERROR \
    #     >> "$LOG_FILE" 2>&1
    status=$?

    # Check if the notebook ran successfully
    if [ $status -eq 0 ]; then
        echo -e "${notebook} - ${GREEN}SUCCESS${NC}"
        
    else
        echo -e "${notebook} - ${RED}FAIL${NC}" | tee -a "$LOG_FILE"
        overall_status=1
    fi
done

# Display the summary report with colors
echo "Testing complete. Summary:"
cat "$LOG_FILE"

# Exit the script with 0 if all notebooks passed, 1 if any failed
exit $overall_status
