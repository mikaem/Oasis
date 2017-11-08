import pytest
import subprocess
import re

number = "([0-9]+.[0-9]+e[+-][0-9]+)"


def test_single_IPCS():
    d = subprocess.check_output("cd ..;mpirun -np 1 python NSfracStep.py problem=TaylorGreen2D T=0.01 Nx=50 Ny=50; cd tests", shell=True)
    match = re.search("Final Error: u0=" + number +
                      " u1=" + number + " p=" + number, str(d))
    err = match.groups()
    for e in err:
        assert eval(e) < 1e-5

    # Make sure the optimized version gives the same result as naive
    d2 = subprocess.check_output("cd ..;mpirun -np 1 python NSfracStep.py solver=IPCS problem=TaylorGreen2D T=0.01 Nx=50 Ny=50; cd tests", shell=True)
    match2 = re.search("Final Error: u0=" + number +
                       " u1=" + number + " p=" + number, str(d2))
    err2 = match2.groups()
    for e1, e2 in zip(err, err2):
        assert abs(eval(e1) - eval(e2)) < 1e-9


def test_mpi_IPCS():
    d = subprocess.check_output("cd ..;mpirun -np 2 python NSfracStep.py problem=TaylorGreen2D T=0.01 Nx=50 Ny=50; cd tests", shell=True)
    match = re.search("Final Error: u0=" + number +
                      " u1=" + number + " p=" + number, str(d))
    err = match.groups()
    for e in err:
        assert eval(e) < 1e-5

    # Make sure the optimized version gives the same result as naive
    d2 = subprocess.check_output("cd ..;mpirun -np 1 python NSfracStep.py solver=IPCS problem=TaylorGreen2D T=0.01 Nx=50 Ny=50; cd tests", shell=True)
    match2 = re.search("Final Error: u0=" + number +
                       " u1=" + number + " p=" + number, str(d2))
    err2 = match2.groups()
    for e1, e2 in zip(err, err2):
        assert abs(eval(e1) - eval(e2)) < 1e-9


def test_single_IPCS2():
    d = subprocess.check_output("cd ..;mpirun -np 1 python NSfracStep.py problem=DrivenCavity T=0.01 Nx=20 Ny=20 plot_interval=10000 testing=True; cd tests", shell=True)
    match = re.search("Velocity norm = " + number, str(d))
    err = match.groups()

    # Make sure the optimized version gives the same result as naive
    d2 = subprocess.check_output("cd ..;mpirun -np 1 python NSfracStep.py solver=IPCS problem=DrivenCavity T=0.01 Nx=20 Ny=20 plot_interval=10000 testing=True; cd tests", shell=True)
    match2 = re.search("Velocity norm = " + number, str(d2))
    err2 = match2.groups()
    assert abs(eval(err[0]) - eval(err2[0])) < 1e-9


def test_mpi_IPCS2():
    d = subprocess.check_output("cd ..;mpirun -np 2 python NSfracStep.py problem=DrivenCavity T=0.01 Nx=20 Ny=20 plot_interval=10000 testing=True; cd tests", shell=True)
    match = re.search("Velocity norm = " + number, str(d))
    err = match.groups()

    # Make sure the optimized version gives the same result as naive
    d2 = subprocess.check_output("cd ..;mpirun -np 1 python NSfracStep.py solver=IPCS problem=DrivenCavity T=0.01 Nx=20 Ny=20 plot_interval=10000 testing=True; cd tests", shell=True)
    match2 = re.search("Velocity norm = " + number, str(d2))
    err2 = match2.groups()
    assert abs(eval(err[0]) - eval(err2[0])) < 1e-9


def test_single_BDFPC():
    d = subprocess.check_output(
        "cd ..;mpirun -np 1 python NSfracStep.py problem=TaylorGreen2D T=0.01 Nx=50 Ny=50 solver=BDFPC; cd tests", shell=True)
    match = re.search("Final Error: u0=" + number +
                      " u1=" + number + " p=" + number, str(d))
    err = match.groups()
    for e in err[:2]:
        assert eval(e) < 1e-5

    # Make sure the optimized version gives the same result as naive
    d2 = subprocess.check_output("cd ..;mpirun -np 1 python NSfracStep.py solver=IPCS problem=TaylorGreen2D T=0.01 Nx=50 Ny=50 solver=BDFPC_Fast; cd tests", shell=True)
    match = re.search("Final Error: u0=" + number + " u1=" +
                      number + " p=" + number, str(d2))
    err2 = match.groups()
    for e1, e2 in zip(err, err2):
        assert abs(eval(e1) - eval(e2)) < 1e-9


def test_mpi_BDFPC():
    d = subprocess.check_output("cd ..;mpirun -np 4 python NSfracStep.py problem=TaylorGreen2D T=0.01 Nx=50 Ny=50 solver=BDFPC_Fast; cd tests", shell=True)
    match = re.search("Final Error: u0=" + number +
                      " u1=" + number + " p=" + number, str(d))
    err = match.groups()
    for e in err[:2]:
        assert eval(e) < 1e-5

    # Make sure the optimized version gives the same result as naive
    d2 = subprocess.check_output("cd ..;mpirun -np 1 python NSfracStep.py problem=TaylorGreen2D T=0.01 Nx=50 Ny=50 solver=BDFPC; cd tests", shell=True)
    match = re.search("Final Error: u0=" + number + " u1=" +
                      number + " p=" + number, str(d2))
    err2 = match.groups()
    for e1, e2 in zip(err, err2):
        assert abs(eval(e1) - eval(e2)) < 1e-9
