import pytest
import subprocess
import re

########
# These tests only check if the untested code runs without breaking, and does not
# test any properties. These should be replace with better tests.
########

@pytest.mark.skip(reason="Time")
@pytest.mark.parametrize("num_p", [1, 2])
@pytest.mark.parametrize("solver", ["IPCS_ABCN", "IPCS_ABE", "Chorin", "BDFPC", "BDFPC_Fast"])
@pytest.mark.parametrize("les_model", ["Smagorinsky", "Wale",
                                       "DynamicLagrangian",
                                       "ScaleDepDynamicLagrangian"])
def test_LES(num_p, solver, les_model):
    cmd = "mpirun -np {} oasis NSfracStep solver={} T=0.0001 dt=0.00005 les_model={}"
    subprocess.check_output(cmd.format(num_p, solver, les_model), shell=True)


@pytest.mark.skip(reason="Time")
@pytest.mark.parametrize("num_p", [1, 2])
@pytest.mark.parametrize("solver", ["IPCS_ABCN", "IPCS_ABE", "Chorin", "BDFPC", "BDFPC_Fast"])
@pytest.mark.parametrize("problem", ["Channel", "Cylinder", "DrivenCavity",
                                     "DrivenCavity3D", "FlowPastSphere3D",
                                     "LaminarChannel", "Lshape", "Skewed2D",
                                     "SkewedFlow", "TaylorGreen2D", "TaylorGreen3D"])
def test_demo_NSfracStep(num_p, solver, problem):
    if problem in ["FlowPastSphere3D", "Skewed2D"]:
        pytest.xfail("Dependent on gmsh")

    if num_p == 2 and problem in ["SkewedFlow", "FlowPastSphere", "Lshape"]:
        pytest.xfail("Submesh does not run in parallell yet")

    cmd = "mpirun -np {} oasis NSfracStep solver={} T=0.0001 dt=0.00005 problem={}"
    subprocess.check_output(cmd.format(num_p, solver, problem), shell=True)

@pytest.mark.skip(reason="Time")
@pytest.mark.parametrize("num_p", [1, 2])
@pytest.mark.parametrize("solver", ["default", "naive"])
@pytest.mark.parametrize("problem", ["Cylinder", "DrivenCavity", "Skewed2D", "Nozzle2D"])
def test_demo_NSCoupled(num_p, solver, problem):
    if problem in ["Skewed2D"]:
        pytest.xfail("Dependent on gmsh")

    if num_p == 2 and problem in ["SkewedFlow"]:
        pytest.xfail("Submesh does not run in parallell yet")

    cmd = "mpirun -np {} oasis NSCoupled solver={} problem={}"
    subprocess.check_output(cmd.format(num_p, solver, problem), shell=True)


if __name__ == "__main__":
    test_demo_NSCoupled()
    test_demo_NSfracStep()
    test_LES()
