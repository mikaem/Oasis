#!/usr/bin/env python

import sys

def main():
    if 'NSfracStep' in sys.argv[1:]:
        print(sys.argv)
        from oasis import NSfracStep

    elif 'NSCoupled' in sys.argv[1:]:
        from oasis import NSCoupled

    else:
        raise NotImplementedError

if __name__ == '__main__':
    main()
