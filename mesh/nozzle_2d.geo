// variables, all in meters
start = -0.18296;
stopp = 0.32;
r1 = 0.006;
r2 = 0.002;
cone_length = 0.022685;
throat_length = 0.04;
cone_start = -throat_length - cone_length;
cone_stopp = - throat_length;
lc = 0.001;

Point(1) = {start, r1,    0, lc };
Point(2) = {cone_start, r1,  0, lc/5 };
Point(3) = {cone_stopp,  r2, 0, lc/5 };
Point(4) = {0,  r2, 0, lc/10 };
Point(5) = {0, r1, 0, lc/10 };
Point(6) = {stopp,  r1,  0, lc };

Point(10) = {start, 0, 0,   lc };
Point(11) = {cone_start, 0, 0, lc/5 };
Point(12) = {cone_stopp, 0, 0, lc/5 };
Point(13) = {0, 0, 0, lc/10};
Point(14) = {stopp, 0, 0,   lc };

Line(1) =  {1, 2};
Line(2) =  {2, 3};
Line(3) =  {3, 4};
Line(4) =  {4, 5};
Line(5) =  {5, 6};
Line(6) =  {6, 14};
Line(7) =  {14,13};
Line(8) =  {13,12};
Line(9) = {12,11};
Line(10) = {11,10};
Line(11) = {10, 1};

Line Loop(30) = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11};
Plane Surface(31) = {30};
DefineConstant[ lc = { 0.1, Path "Gmsh/Parameters"}];
DefineConstant[ lc = { 0.1, Path "Gmsh/Parameters"}];
Extrude {{1, 0, 0}, {0, 0, 0}, Pi} {
  Surface{31};
}
