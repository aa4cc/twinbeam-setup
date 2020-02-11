function [Erz, Erz_cplx] = rsBackPropMud_fun( img, z, dx, n, lambda )
% Implemented according to:
% [1]O. Mudanyali, D. Tseng, C. Oh, S. O. Isikman, I. Sencan, W. Bishara, C. Oztoprak, S. Seo, B. Khademhosseini, and A. Ozcan, “Compact, Light-weight and Cost-effective Microscope based on Lensless Incoherent Holography for Telemedicine Applications,” Lab Chip, vol. 10, no. 11, pp. 1417–1428, Jun. 2010.
% [2]S.-H. Lee and D. G. Grier, “Holographic microscopy of holographically trapped three-dimensional structures,” Optics express, vol. 15, no. 4, pp. 1505–1512, 2007.
% [3]F. C. Cheong, B. J. Krishnatreya, and D. G. Grier, “Strategies for three-dimensional particle tracking with holographic video microscopy,” Opt. Express, vol. 18, no. 13, pp. 13563–13573, 2010.
% Note: In [1] They use spatial frequency and in [2,3] spatial angular
% frequency
%
% The derivation of the formulae for the propagator (without refractive
% index n) can be found in (p. 60):
% [3]J. W. Goodman, Introduction to Fourier optics. McGraw-Hill, 1996.


img = double(img);
[M, N] = size(img);

lM = dx*M;
lN = dx*N;
fx=-1/(2*dx):1/lN:1/(2*dx)-1/lN;
fy=-1/(2*dx):1/lM:1/(2*dx)-1/lM;
[FX, FY] = meshgrid(fx, fy);

% Rayleigh-Sommerfeld
Hq = exp(1i*2*pi*z*n/lambda*sqrt(1-(lambda*FX/n).^2 -(lambda*FY/n).^2));
Hq = Hq.*(sqrt(FX.^2+FY.^2) < (n/lambda));

Hq = fftshift(Hq);

Bq = fft2(img);
Erz_cplx = Hq.*Bq;
Erz_cplx = ifft2(Erz_cplx);
Erz = abs(Erz_cplx);

end

