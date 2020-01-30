#!/bin/sh -e
sysctl -w kernel.sched_rt_runtime_us=-1
exit 0