import pytest
import subprocess
import re
import platform

number = "[+-]([0-9]+.[0-9]+e[+-][0-9]+)"


def test_default_Coupled():
    d = subprocess.check_output("mpirun -np 1 oasis NSCoupled problem=Cylinder testing=True", shell=True)
    match = re.search("Cd = "+number+", CL = "+number, str(d))
    err = match.groups(0)
    assert round(eval(err[0]) - 5.5739206, 6) == 0
    assert round(eval(err[1]) - 0.0107497, 6) == 0


#@pytest.mark.skip(reason="Time")
def test_default_CR_Coupled():
    d = subprocess.check_output("mpirun -np 1 oasis NSCoupled problem=Cylinder testing=True element=CR", shell=True)
    match = re.search("Cd = "+number+", CL = "+number, str(d))
    err = match.groups(0)
    assert round(eval(err[0]) - 5.587641, 6) == 0
    assert round(eval(err[1]) - 0.0116757, 6) == 0


@pytest.mark.skip(reason="Fenics fails on enriched element")
def test_default_MINI_Coupled():
    d = subprocess.check_output("mpirun -np 1 oasis NSCoupled problem=Cylinder testing=True element=MINI", shell=True)
    match = re.search("Cd = "+number+", CL = "+number, str(d))
    err = match.groups(0)
    assert round(eval(err[0]) - 5.534679, 5) == 0
    assert round(eval(err[1]) - 0.0102132, 5) == 0


#@pytest.mark.skip(reason="Time")
def test_naive_Coupled():
    d = subprocess.check_output("mpirun -np 1 oasis NSCoupled problem=Cylinder solver=naive testing=True", shell=True)
    match = re.search("Cd = "+number+", CL = "+number, str(d))
    err = match.groups(0)
    assert round(eval(err[0]) - 5.5739206, 6) == 0
    assert round(eval(err[1]) - 0.0107497, 6) == 0


#@pytest.mark.skipif(platform.system() == "Darwin", reason="Parallel LU solver fails on Darwin")
@pytest.mark.skip(reason="Problem with direct solvers and MPI")
def test_default_mpi_Coupled():
    d = subprocess.check_output("mpirun -np 2 oasis NSCoupled problem=Cylinder testing=True", shell=True)
    match = re.search("Cd = "+number+", CL = "+number, str(d))
    err = match.groups(0)
    assert round(eval(err[0]) - 5.5739206, 6) == 0
    assert round(eval(err[1]) - 0.0107497, 6) == 0

#@pytest.mark.skipif(platform.system() == "Darwin", reason="Parallel LU solver fails on Darwin")
@pytest.mark.skip(reason="Problem with direct solvers and MPI")
def test_naive_mpi_Coupled():
    d = subprocess.check_output("mpirun -np 2 oasis NSCoupled problem=Cylinder solver=naive testing=True", shell=True)
    match = re.search("Cd = "+number+", CL = "+number, str(d))
    err = match.groups(0)
    assert round(eval(err[0]) - 5.5739206, 6) == 0
    assert round(eval(err[1]) - 0.0107497, 6) == 0

if __name__ == '__main__':
    test_default_Coupled()
    #test_default_CR_Coupled()
    #test_default_MINI_Coupled()
    #test_naive_Coupled()
