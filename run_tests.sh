#!/usr/bin/env bash

source venv_fewerbytes/bin/activate

echo "Executing tests..."
echo ""
coverage erase


coverage run -a --omit "venv_fewerbytes/*" -m tests.test_types
coverage run -a --omit "venv_fewerbytes/*" -m tests.test_integer_compression

report_coverage=false
include_missing=false
for i in "$@"
do
case $i in
    r|-r)
    report_coverage=true
    shift
    ;;
    m|-m)
    include_missing=true
    shift
    ;;
    *)
    shift
    ;;
esac
done

if $report_coverage
then
    echo "Coverage report:"
    if $include_missing
    then
        coverage report -m
    else
        coverage report
    fi
    echo "End of coverage report."
fi

deactivate
echo "Tests completed."