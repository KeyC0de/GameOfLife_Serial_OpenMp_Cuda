#!/bin/bash

export EXTRAE_HOME=/usr/local/extrae
export EXTRAE_CONFIG_FILE=../extrae.xml
export LD_PRELOAD=$EXTRAE_HOME/lib/libmpitrace.so
export EXE=$2 # $2: second argument given by terminal
export TRACENAME=${EXE}.prv

$*
