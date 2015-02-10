__author__ = 'Joakim Boe <joakim.bo@mn.uio.no>'
__date__ = '2015-02-04'
__copyright__ = 'Copyright (C) 2015 ' + __author__
__license__  = 'GNU Lesser GPL version 3 or any later version'

from dolfin import Function

__all__ = ['boussinesq_setup', 'boussinesq_update']

def boussinesq_setup(boussinesq, V, x_1, scalar_components, mesh, M, b0, **NS_namespace):
    
    # Check if boussinesq is to be applied, if not break function
    if boussinesq["use"] == False:
        bouss_code = "pass"
        return dict(Temp=None, bouss_code=bouss_code)

    # Exctract values from boussinesq dict
    g = boussinesq["g"]             # Gravity
    T_ref = boussinesq["T_ref"]     # Reference temp
    beta = boussinesq["beta"]       # Effect of temp diff.
    vertdir = boussinesq["vertical_direction"]   # Vertical direction
    T_index = scalar_components[boussinesq["Temp_scalar_index"]]
    boussinesq.update(T_index=T_index)
    
    # Create function for holding temperature vector and update its values
    Temp = Function(V)
    Temp.vector().axpy(1, x_1[T_index])
    Temp.vector().set_local(-g + beta*(Temp.vector().array()-T_ref))
    Temp.vector().apply("insert")

    # Check if vertical direction has been specified, if not assume y in 2D
    # and z in 3D
    if vertdir == None:
        if mesh.topology().dim() == 2:
            veldir = "u1"
        elif mesh.topology().dim() == 3:
            veldir = "u2"
    # Else vertical direction has been specified as either x, y or z
    elif vertdir == "x":
        veldir = "u0"
    elif vertdir == "y":
        veldir = "u1"
    elif vertdir == "z":
        veldir = "u2"

    boussinesq.update(veldir=veldir)

    # Set up bouss code applying the matrix vector product
    b0[veldir] += M*Temp.vector()
    bouss_code = """
b = boussinesq

Temp.vector().zero()
Temp.vector().axpy(1, x_1[b["T_index"]])
Temp.vector().set_local(-b["g"] + b["beta"]*(Temp.vector().array()-b["T_ref"]))
Temp.vector().apply("insert")
b0[b["veldir"]] += M*Temp.vector()
"""
    return dict(Temp=Temp, bouss_code=bouss_code)

def boussinesq_update(boussinesq, bouss_code, Temp, b0, x_1, M, **NS_namespace):
    exec(bouss_code)
