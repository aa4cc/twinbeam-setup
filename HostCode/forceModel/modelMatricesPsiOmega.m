function [ Psi, Omega ] = modelMatricesPsiOmega( p )

Psi = cell(3,1);
Omega = cell(3,1);

k_DEP = 3.47538687303371e-23; %# pi*obj.e_0*obj.Medium.Permittivity*obj.Particle.Radius^3
f_CM_R = -0.461847136009005;
f_CM_I = -0.145447673066018;
DEP_ampl = 16;

[ Gamma, Lambda_x, Lambda_y, Lambda_z] = modelMatricesGammaLambda( p );

Psi{1} = 2*DEP_ampl^2*k_DEP*f_CM_R*Lambda_x*Gamma';
Psi{2} = 2*DEP_ampl^2*k_DEP*f_CM_R*Lambda_y*Gamma';
Psi{3} = 2*DEP_ampl^2*k_DEP*f_CM_R*Lambda_z*Gamma';

Omega{1} = 2*DEP_ampl^2*k_DEP*f_CM_I*( Gamma*Lambda_x' - Lambda_x*Gamma' );
Omega{2} = 2*DEP_ampl^2*k_DEP*f_CM_I*( Gamma*Lambda_y' - Lambda_y*Gamma' );
Omega{3} = 2*DEP_ampl^2*k_DEP*f_CM_I*( Gamma*Lambda_z' - Lambda_z*Gamma' );

end

