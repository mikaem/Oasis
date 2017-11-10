#!/usr/bin/env python

import sys

def main():
    if 'NSfracStep' in sys.argv[1:]:
        from oasis import NSfracStep

    elif 'NSCoupled' in sys.argv[1:]:
        from oasis import NSCoupled

    else:
        raise NotImplementedError

