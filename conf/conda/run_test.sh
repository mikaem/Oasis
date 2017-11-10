#!/bin/bash
set -e

if [[ "$(uname)" == "Darwin" ]]; then
  export MACOSX_DEPLOYMENT_TARGET=10.9
  export CXXFLAGS="-std=c++11 -stdlib=libc++ $CXXFLAGS"
fi

python -b -m pytest -vs tests/

#mpirun -np 2 py.test -v
