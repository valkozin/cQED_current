#!/bin/bash
pkill -f "python3 run_nonadiabatic"
sleep 2
pgrep -fc "python3 run_nonadiabatic"
