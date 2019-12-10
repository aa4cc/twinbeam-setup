function [ Gamma, Lambda_x, Lambda_y, Lambda_z] = modelMatricesGammaLambda( p )

elNum = 56;

a = [0.17085589111184;0.159430457039006;0.103698822318765;0.0786649226234773;0.0595485428423376;0.0584822518412492;0.0631496109296414;0.0719777663342492;0.110103759311239;0.124087975648196];
b = [1.07665878791643;1.24810401255122;1.44491014565381;1.70480215059619;1.91782224818094;2.07785046161065;2.26622039595283;2.46774228797269;2.68178041561257;2.91999997038365];

Ex = zeros(elNum,1);
Ey = zeros(elNum,1);
Ez = zeros(elNum,1);
Psixx = zeros(elNum,1);
Psiyy = zeros(elNum,1);
Psizz = zeros(elNum,1);
Psixy = zeros(elNum,1);
Psixz = zeros(elNum,1);
Psiyz = zeros(elNum,1);

for k = 1:elNum
    [ ~, Ex(k), Ey(k), Ez(k), Psixx(k), Psixy(k), Psixz(k), Psiyy(k), Psiyz(k), Psizz(k) ] = potDeriv_oneEl( k, p(1), p(2), p(3), a, b );
end

Gamma       = [Ex Ey Ez];
Lambda_x    = [Psixx Psixy Psixz];
Lambda_y    = [Psixy Psiyy Psiyz];
Lambda_z    = [Psixz Psiyz Psizz];

end

