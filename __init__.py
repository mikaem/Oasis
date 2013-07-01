"""
Optimized And StrIpped Solvers
"""
from NSdefault_hooks import *

from commands import getoutput
import time, copy

def getMyMemoryUsage():
    mypid = getpid()
    mymemory = getoutput("ps -o rss %s" % mypid).split()[1]
    return mymemory

# Convenience functions
def strain(u):
    return 0.5*(grad(u)+ grad(u).T)

def omega(u):
    return 0.5*(grad(u) - grad(u).T)

def Omega(u):
    return inner(omega(u), omega(u))

def Strain(u):
    return inner(strain(u), strain(u))

def QC(u):
    return Omega(u) - Strain(u)
            
# The following helper functions are available in dolfin
# They are redefined here for printing only on process 0. 
RED   = "\033[1;37;31m%s\033[0m"
BLUE  = "\033[1;37;34m%s\033[0m"
GREEN = "\033[1;37;32m%s\033[0m"

def info_blue(s, check=True):
    if MPI.process_number()==0 and check:
        print BLUE % s

def info_green(s, check=True):
    if MPI.process_number()==0 and check:
        print GREEN % s
    
def info_red(s, check=True):
    if MPI.process_number()==0 and check:
        print RED % s
