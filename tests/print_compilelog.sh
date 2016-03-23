#!/bin/bash

for f in *; do 
  if [ -d ${f} ]; then
    echo $f
    cat $f/compile.log
  fi
done