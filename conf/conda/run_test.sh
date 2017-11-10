#!/bin/bash
set -e

if [[ "$(uname)" == "Darwin" ]]; then
  export MACOSX_DEPLOYMENT_TARGET=10.9
  export CXXFLAGS="-std=c++11 -stdlib=libc++ $CXXFLAGS"
fi

export INSTANT_CACHE_DIR="${SRC_DIR}/instant"

pushd "$SRC_DIR/tests"
python -b -m pytest -vs

#mpirun -np 2 py.test -v
