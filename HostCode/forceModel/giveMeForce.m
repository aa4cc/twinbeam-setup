function [ Fx, Fy, Fz] = giveMeForce( x, y, z, u )

K =  -0.461847136009005 - 0.1454476730660181i;
Fvz = -3.339997057833991e-11;
r = 2.500000000000000e-05;  
e0 = 8.850000000000001e-12;
em = 80;

% !!! NOT sure about the multiplication by two !!!!
k_r = 2*pi*e0*em*r^3*real(K);
k_i = 2*pi*e0*em*r^3*imag(K);

[ Gamma, Lambda_x, Lambda_y, Lambda_z] = modelMatrices( x, y, z );

Ax = k_r*(Lambda_x*Gamma' + Gamma*Lambda_x') + 1i*k_i*(Lambda_x*Gamma' - Gamma*Lambda_x');
Ay = k_r*(Lambda_y*Gamma' + Gamma*Lambda_y') + 1i*k_i*(Lambda_y*Gamma' - Gamma*Lambda_y');
Az = k_r*(Lambda_z*Gamma' + Gamma*Lambda_z') + 1i*k_i*(Lambda_z*Gamma' - Gamma*Lambda_z');

Fx = u'*Ax*u;
Fy = u'*Ay*u;
Fz = u'*Az*u+Fvz;

end

