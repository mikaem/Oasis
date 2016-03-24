import pytest
import subprocess, re

number = "[+-]([0-9]+.[0-9]+e[+-][0-9]+)"

def test_default_Coupled():
    d = subprocess.check_output("cd ..;mpirun -np 1 python NSCoupled.py problem=Cylinder testing=True; cd tests", shell=True)
    match = re.search("Cd = "+number+", CL = "+number, d)
    err = match.groups(0)
    assert round(eval(err[0]) - 5.5739206, 6) == 0
    assert round(eval(err[1]) - 0.0107497, 6) == 0

def test_default_CR_Coupled():
    d = subprocess.check_output("cd ..;mpirun -np 1 python NSCoupled.py problem=Cylinder testing=True element=CR; cd tests", shell=True)
    match = re.search("Cd = "+number+", CL = "+number, d)
    err = match.groups(0)
    assert round(eval(err[0]) - 5.587641, 6) == 0
    assert round(eval(err[1]) - 0.0116757, 6) == 0
    
def test_default_MINI_Coupled():
    d = subprocess.check_output("cd ..;mpirun -np 1 python NSCoupled.py problem=Cylinder testing=True element=MINI; cd tests", shell=True)
    match = re.search("Cd = "+number+", CL = "+number, d)
    err = match.groups(0)
    assert round(eval(err[0]) - 5.534679, 5) == 0
    assert round(eval(err[1]) - 0.0102132, 5) == 0

def test_naive_Coupled():
    d = subprocess.check_output("cd ..;mpirun -np 1 python NSCoupled.py problem=Cylinder solver=naive testing=True; cd tests", shell=True)
    match = re.search("Cd = "+number+", CL = "+number, d)
    err = match.groups(0)
    assert round(eval(err[0]) - 5.5739206, 6) == 0
    assert round(eval(err[1]) - 0.0107497, 6) == 0

def test_default_mpi_Coupled():
    d = subprocess.check_output("cd ..;mpirun -np 2 python NSCoupled.py problem=Cylinder testing=True; cd tests", shell=True)
    match = re.search("Cd = "+number+", CL = "+number, d)
    err = match.groups(0)
    assert round(eval(err[0]) - 5.5739206, 6) == 0
    assert round(eval(err[1]) - 0.0107497, 6) == 0

def test_naive_mpi_Coupled():
    d = subprocess.check_output("cd ..;mpirun -np 2 python NSCoupled.py problem=Cylinder solver=naive testing=True; cd tests", shell=True)
    match = re.search("Cd = "+number+", CL = "+number, d)
    err = match.groups(0)
    assert round(eval(err[0]) - 5.5739206, 6) == 0
    assert round(eval(err[1]) - 0.0107497, 6) == 0

def test_cylindrical_Coupled():
    d = subprocess.check_output("cd ..;mpirun -np 1 python NSCoupled.py problem=Pipe solver=cylindrical testing=True; cd tests", shell=True)
