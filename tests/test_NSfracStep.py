import pytest
import subprocess
import re
import math

number = "([0-9]+.[0-9]+e[+-][0-9]+)"


@pytest.mark.parametrize("solver", ["IPCS_ABCN", "BDFPC_Fast"])
@pytest.mark.parametrize("num_p", [1])
def test_spatial_rate_of_convergence(num_p, solver):
    cmd = ("mpirun -np {} oasis NSfracStep solver={} "
            "problem=TaylorGreen2D compute_error=1e8 T={} dt={} Nx={} Ny={}")
    p_err = []
    u0_err = []
    dt = 0.0001
    T = dt*4
    N = [20, 26, 32, 38]
    #N = [20, 30, 40, 50, 60]

    for n in N:
        d = subprocess.check_output(cmd.format(num_p, solver, T, dt, n, n),
                                    shell=True)
        match = re.search("Final Error: u0=" + number + " u1="
                          + number + " p=" + number, str(d))
        err = match.groups()
        u0_err.append(eval(err[0]))
        p_err.append(eval(err[2]))

    p_conv = []
    u_conv = []
    dx = [math.sqrt(2*(1./n)**2) for n in N]
    for i in range(len(dx)-1):
        u_conv.append(math.log(u0_err[i] / u0_err[i+1]) / math.log(dx[i] / dx[i+1]))
        p_conv.append(math.log(p_err[i] / p_err[i+1]) / math.log(dx[i] / dx[i+1]))

    assert round(u_conv[-1], 1) == 2.0
    # FIXME: Doublecheck that it makes sence that p converges as dx**4
    assert round(p_conv[-1], 1) == 4.0


# FIXME: Should add a working temporal convergence as well
"""
@pytest.mark.parametrize("solver", ["IPCS"])
@pytest.mark.parametrize("num_p", [1, 4])
def test_temporal_rate_of_convergence(num_p, solver):
    cmd = ("cd ..; mpirun -np {} python NSfracStep.py solver={} "
           "problem=TaylorGreen2D T={} dt={} Nx={} Ny={} velocity_degree={} "
           "pressure_degree={}; cd tests")
    p_err = []
    u0_err = []
    DT = [0.003, 0.004, 0.006]
    T = 0.012
    N = 100
    u_deg = 5
    p_deg = 5

    for dt in DT:
        d = subprocess.check_output("ls")
        d = subprocess.check_output(cmd.format(num_p, solver, T, dt, N, N,
                                               u_deg, p_deg), shell=True)
        match = re.search("Final Error: u0=" + number + " u1="
                          + number + " p=" + number, str(d))
        err = match.groups()
        u0_err.append(eval(err[0]))
        p_err.append(eval(err[2]))

    p_conv = []
    u_conv = []
    for i in range(len(DT)-1):
        u_conv.append(math.log(u0_err[i] / u0_err[i+1]) / math.log(DT[i] / DT[i+1]))
        p_conv.append(math.log(p_err[i] / p_err[i+1]) / math.log(DT[i] / DT[i+1]))

    #print(u_conv) [1.5895713782111356, 1.2879429547093262]
    #print(p_conv) [1.989093812226882, 1.9729918495191832]
    assert round(u_conv[-1], 2) == 2.0
    assert round(p_conv[-1], 1) == 4.0
"""


@pytest.mark.parametrize("solver", ["IPCS_ABCN", "IPCS_ABE", "Chorin", "BDFPC_Fast"])
@pytest.mark.parametrize("num_p", [1, 2])
def test_TaylorGreen2D(solver, num_p):
    cmd = ("mpirun -np {} oasis NSfracStep solver={} "
           "problem=TaylorGreen2D T=0.01 Nx=40 Ny=40")
    if num_p > 1 and solver == "Chorin":  # Uses direct solver
        return

    d = subprocess.check_output(cmd.format(num_p, solver), shell=True)
    match = re.search("Final Error: u0=" + number +
                      " u1=" + number + " p=" + number, str(d))
    err = match.groups()
    if "IPCS_AB" in solver:
        for e in err:
            assert eval(e) < 1e-4
    elif "Chorin" in solver:
        for e in err[:2]:
            assert eval(e) < 1e-4
    else:
        for e in err[:2]:
            assert eval(e) < 1e-4

    # Make sure the optimized version gives the same result as naive
    slow=None
    if "IPCS_AB" in solver:
        slow="IPCS"
    elif "BDFPC" in solver:
        slow="BDFPC"
    if not slow is None:
        d2 = subprocess.check_output(cmd.format(1, slow), shell=True)
        match2 = re.search("Final Error: u0=" + number +
                           " u1=" + number + " p=" + number, str(d2))
        err2 = match2.groups()

    if "BDFPC" in solver:
        for e1, e2 in zip(err[:2], err2[:2]):
            assert abs(eval(e1) - eval(e2)) < 1e-9
    elif "IPCS_AB" in solver:
        for e1, e2 in zip(err[:2], err2[:2]):
            assert abs(eval(e1) - eval(e2)) < 1e-8


@pytest.mark.parametrize("num_p", [1, 2])
def test_DrivenCavity(num_p):
    cmd = ("mpirun -np {} oasis NSfracStep problem=DrivenCavity T=0.01 "
          "Nx=20 Ny=20 plot_interval=10000 solver={} testing=True")
    d = subprocess.check_output(cmd.format(num_p, "IPCS_ABCN"), shell=True)
    match = re.search("Velocity norm = " + number, str(d))
    err = match.groups()

    # Make sure the optimized version gives the same result as naive
    d2 = subprocess.check_output(cmd.format(1, "IPCS"), shell=True)
    match2 = re.search("Velocity norm = " + number, str(d2))
    err2 = match2.groups()
    assert abs(eval(err[0]) - eval(err2[0])) < 1e-9

if __name__ == '__main__':
    #test_DrivenCavity()
    #test_TaylorGreen2D()
    #test_spatial_rate_of_convergence(1, "BDFPC_Fast")
    test_spatial_rate_of_convergence(1, "IPCS_ABCN")
    #test_temporal_rate_of_convergence()
