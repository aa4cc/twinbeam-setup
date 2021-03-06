function [ phi, Ex, Ey, Ez, Psixx, Psixy, Psixz, Psiyy, Psiyz, Psizz ] = potDeriv_baseEl( x, y, z, d )


phi=(1/2).*pi.^(-1).*(atan(((1/2).*d+(-1).*y).*z.^(-1))+atan(((1/2).*d+y).* ...
  z.^(-1))+atan(x.*((-1/2).*d+y).*z.^(-1).*(x.^2+((-1/2).*d+y).^2+z.^2).^( ...
  -1/2))+(-1).*atan(x.*((1/2).*d+y).*z.^(-1).*(x.^2+((1/2).*d+y).^2+z.^2) ...
  .^(-1/2)));

Ex = (1/2).*pi.^(-1).*z.*(x.^2+z.^2).^(-1).*(2.*y.*((-1).*(d.^2+(-4).*d.*y+ ...
  4.*(x.^2+y.^2+z.^2)).^(-1/2)+(d.^2+4.*d.*y+4.*(x.^2+y.^2+z.^2)).^(-1/2)) ...
  +d.*((d.^2+(-4).*d.*y+4.*(x.^2+y.^2+z.^2)).^(-1/2)+(d.^2+4.*d.*y+4.*( ...
  x.^2+y.^2+z.^2)).^(-1/2)));

Ey = (-1/2).*pi.^(-1).*((-4).*z.*(d.^2+(-4).*d.*y+4.*(y.^2+z.^2)).^(-1)+4.* ...
  z.*(d.^2+4.*d.*y+4.*(y.^2+z.^2)).^(-1)+8.*x.*z.*(d.^2+(-4).*d.*y+4.*( ...
  y.^2+z.^2)).^(-1).*(d.^2+(-4).*d.*y+4.*(x.^2+y.^2+z.^2)).^(-1/2)+(-8).* ...
  x.*z.*(d.^2+4.*d.*y+4.*(y.^2+z.^2)).^(-1).*(d.^2+4.*d.*y+4.*(x.^2+y.^2+ ...
  z.^2)).^(-1/2));

Ez = (-1/2).*pi.^(-1).*(((-2).*d+4.*y).*(d.^2+(-4).*d.*y+4.*(y.^2+z.^2)).^( ...
  -1)+(-2).*(d+2.*y).*(d.^2+4.*d.*y+4.*(y.^2+z.^2)).^(-1)+x.*(d+(-2).*y).* ...
  (x.^2+z.^2).^(-1).*(d.^2+(-4).*d.*y+4.*(y.^2+z.^2)).^(-1).*(d.^2+(-4).* ...
  d.*y+4.*(x.^2+y.^2+z.^2)).^(-1/2).*(d.^2+(-4).*d.*y+4.*(x.^2+y.^2+2.* ...
  z.^2))+x.*(d+2.*y).*(x.^2+z.^2).^(-1).*(d.^2+4.*d.*y+4.*(y.^2+z.^2)).^( ...
  -1).*(d.^2+4.*d.*y+4.*(x.^2+y.^2+z.^2)).^(-1/2).*(d.^2+4.*d.*y+4.*(x.^2+ ...
  y.^2+2.*z.^2)));

Psixx = (1/2).*pi.^(-1).*z.*(x.^2+z.^2).^(-1).*(8.*x.*y.*((d.^2+(-4).*d.*y+4.*( ...
  x.^2+y.^2+z.^2)).^(-3/2)+(-1).*(d.^2+4.*d.*y+4.*(x.^2+y.^2+z.^2)).^( ...
  -3/2))+d.*((-4).*x.*(d.^2+(-4).*d.*y+4.*(x.^2+y.^2+z.^2)).^(-3/2)+(-4).* ...
  x.*(d.^2+4.*d.*y+4.*(x.^2+y.^2+z.^2)).^(-3/2)))+(-1).*pi.^(-1).*x.*z.*( ...
  x.^2+z.^2).^(-2).*(2.*y.*((-1).*(d.^2+(-4).*d.*y+4.*(x.^2+y.^2+z.^2)).^( ...
  -1/2)+(d.^2+4.*d.*y+4.*(x.^2+y.^2+z.^2)).^(-1/2))+d.*((d.^2+(-4).*d.*y+ ...
  4.*(x.^2+y.^2+z.^2)).^(-1/2)+(d.^2+4.*d.*y+4.*(x.^2+y.^2+z.^2)).^(-1/2)) ...
  );

Psixy = (1/2).*pi.^(-1).*z.*(x.^2+z.^2).^(-1).*(d.*(2.*(d+(-2).*y).*(d.^2+(-4).* ...
  d.*y+4.*(x.^2+y.^2+z.^2)).^(-3/2)+(-2).*(d+2.*y).*(d.^2+4.*d.*y+4.*( ...
  x.^2+y.^2+z.^2)).^(-3/2))+2.*y.*(((-2).*d+4.*y).*(d.^2+(-4).*d.*y+4.*( ...
  x.^2+y.^2+z.^2)).^(-3/2)+(-2).*(d+2.*y).*(d.^2+4.*d.*y+4.*(x.^2+y.^2+ ...
  z.^2)).^(-3/2))+2.*((-1).*(d.^2+(-4).*d.*y+4.*(x.^2+y.^2+z.^2)).^(-1/2)+ ...
  (d.^2+4.*d.*y+4.*(x.^2+y.^2+z.^2)).^(-1/2)));

Psixz = (1/2).*pi.^(-1).*z.*(x.^2+z.^2).^(-1).*(8.*y.*z.*((d.^2+(-4).*d.*y+4.*( ...
  x.^2+y.^2+z.^2)).^(-3/2)+(-1).*(d.^2+4.*d.*y+4.*(x.^2+y.^2+z.^2)).^( ...
  -3/2))+d.*((-4).*z.*(d.^2+(-4).*d.*y+4.*(x.^2+y.^2+z.^2)).^(-3/2)+(-4).* ...
  z.*(d.^2+4.*d.*y+4.*(x.^2+y.^2+z.^2)).^(-3/2)))+(-1).*pi.^(-1).*z.^2.*( ...
  x.^2+z.^2).^(-2).*(2.*y.*((-1).*(d.^2+(-4).*d.*y+4.*(x.^2+y.^2+z.^2)).^( ...
  -1/2)+(d.^2+4.*d.*y+4.*(x.^2+y.^2+z.^2)).^(-1/2))+d.*((d.^2+(-4).*d.*y+ ...
  4.*(x.^2+y.^2+z.^2)).^(-1/2)+(d.^2+4.*d.*y+4.*(x.^2+y.^2+z.^2)).^(-1/2)) ...
  )+(1/2).*pi.^(-1).*(x.^2+z.^2).^(-1).*(2.*y.*((-1).*(d.^2+(-4).*d.*y+4.* ...
  (x.^2+y.^2+z.^2)).^(-1/2)+(d.^2+4.*d.*y+4.*(x.^2+y.^2+z.^2)).^(-1/2))+ ...
  d.*((d.^2+(-4).*d.*y+4.*(x.^2+y.^2+z.^2)).^(-1/2)+(d.^2+4.*d.*y+4.*( ...
  x.^2+y.^2+z.^2)).^(-1/2)));

Psiyy = (-1/2).*pi.^(-1).*(4.*((-4).*d+8.*y).*z.*(d.^2+(-4).*d.*y+4.*(y.^2+z.^2) ...
  ).^(-2)+(-16).*(d+2.*y).*z.*(d.^2+4.*d.*y+4.*(y.^2+z.^2)).^(-2)+16.*x.*( ...
  d+(-2).*y).*z.*(d.^2+(-4).*d.*y+4.*(y.^2+z.^2)).^(-1).*(d.^2+(-4).*d.*y+ ...
  4.*(x.^2+y.^2+z.^2)).^(-3/2)+32.*x.*(d+(-2).*y).*z.*(d.^2+(-4).*d.*y+4.* ...
  (y.^2+z.^2)).^(-2).*(d.^2+(-4).*d.*y+4.*(x.^2+y.^2+z.^2)).^(-1/2)+16.* ...
  x.*(d+2.*y).*z.*(d.^2+4.*d.*y+4.*(y.^2+z.^2)).^(-1).*(d.^2+4.*d.*y+4.*( ...
  x.^2+y.^2+z.^2)).^(-3/2)+32.*x.*(d+2.*y).*z.*(d.^2+4.*d.*y+4.*(y.^2+ ...
  z.^2)).^(-2).*(d.^2+4.*d.*y+4.*(x.^2+y.^2+z.^2)).^(-1/2));

Psiyz = (-1/2).*pi.^(-1).*((-4).*(d.^2+(-4).*d.*y+4.*(y.^2+z.^2)).^(-1)+4.*( ...
  d.^2+4.*d.*y+4.*(y.^2+z.^2)).^(-1)+8.*x.*(d.^2+(-4).*d.*y+4.*(y.^2+z.^2) ...
  ).^(-1).*(d.^2+(-4).*d.*y+4.*(x.^2+y.^2+z.^2)).^(-1/2)+(-8).*x.*(d.^2+ ...
  4.*d.*y+4.*(y.^2+z.^2)).^(-1).*(d.^2+4.*d.*y+4.*(x.^2+y.^2+z.^2)).^( ...
  -1/2)+32.*z.^2.*((d.^2+(-4).*d.*y+4.*(y.^2+z.^2)).^(-2)+(-1).*(d.^2+4.* ...
  d.*y+4.*(y.^2+z.^2)).^(-2)+x.*((-1).*(d.^2+(-4).*d.*y+4.*(y.^2+z.^2)).^( ...
  -1).*(d.^2+(-4).*d.*y+4.*(x.^2+y.^2+z.^2)).^(-3/2)+(-2).*(d.^2+(-4).*d.* ...
  y+4.*(y.^2+z.^2)).^(-2).*(d.^2+(-4).*d.*y+4.*(x.^2+y.^2+z.^2)).^(-1/2)+( ...
  d.^2+4.*d.*y+4.*(y.^2+z.^2)).^(-1).*(d.^2+4.*d.*y+4.*(x.^2+y.^2+z.^2)) ...
  .^(-3/2)+2.*(d.^2+4.*d.*y+4.*(y.^2+z.^2)).^(-2).*(d.^2+4.*d.*y+4.*(x.^2+ ...
  y.^2+z.^2)).^(-1/2))));

Psizz = (-1/2).*pi.^(-1).*(16.*(d+(-2).*y).*z.*(d.^2+(-4).*d.*y+4.*(y.^2+z.^2)) ...
  .^(-2)+16.*(d+2.*y).*z.*(d.^2+4.*d.*y+4.*(y.^2+z.^2)).^(-2)+16.*x.*(d+( ...
  -2).*y).*z.*(x.^2+z.^2).^(-1).*(d.^2+(-4).*d.*y+4.*(y.^2+z.^2)).^(-1).*( ...
  d.^2+(-4).*d.*y+4.*(x.^2+y.^2+z.^2)).^(-1/2)+16.*x.*(d+2.*y).*z.*(x.^2+ ...
  z.^2).^(-1).*(d.^2+4.*d.*y+4.*(y.^2+z.^2)).^(-1).*(d.^2+4.*d.*y+4.*( ...
  x.^2+y.^2+z.^2)).^(-1/2)+(-4).*x.*(d+(-2).*y).*z.*(x.^2+z.^2).^(-1).*( ...
  d.^2+(-4).*d.*y+4.*(y.^2+z.^2)).^(-1).*(d.^2+(-4).*d.*y+4.*(x.^2+y.^2+ ...
  z.^2)).^(-3/2).*(d.^2+(-4).*d.*y+4.*(x.^2+y.^2+2.*z.^2))+(-8).*x.*(d+( ...
  -2).*y).*z.*(x.^2+z.^2).^(-1).*(d.^2+(-4).*d.*y+4.*(y.^2+z.^2)).^(-2).*( ...
  d.^2+(-4).*d.*y+4.*(x.^2+y.^2+z.^2)).^(-1/2).*(d.^2+(-4).*d.*y+4.*(x.^2+ ...
  y.^2+2.*z.^2))+(-2).*x.*(d+(-2).*y).*z.*(x.^2+z.^2).^(-2).*(d.^2+(-4).* ...
  d.*y+4.*(y.^2+z.^2)).^(-1).*(d.^2+(-4).*d.*y+4.*(x.^2+y.^2+z.^2)).^( ...
  -1/2).*(d.^2+(-4).*d.*y+4.*(x.^2+y.^2+2.*z.^2))+(-4).*x.*(d+2.*y).*z.*( ...
  x.^2+z.^2).^(-1).*(d.^2+4.*d.*y+4.*(y.^2+z.^2)).^(-1).*(d.^2+4.*d.*y+4.* ...
  (x.^2+y.^2+z.^2)).^(-3/2).*(d.^2+4.*d.*y+4.*(x.^2+y.^2+2.*z.^2))+(-8).* ...
  x.*(d+2.*y).*z.*(x.^2+z.^2).^(-1).*(d.^2+4.*d.*y+4.*(y.^2+z.^2)).^(-2).* ...
  (d.^2+4.*d.*y+4.*(x.^2+y.^2+z.^2)).^(-1/2).*(d.^2+4.*d.*y+4.*(x.^2+y.^2+ ...
  2.*z.^2))+(-2).*x.*(d+2.*y).*z.*(x.^2+z.^2).^(-2).*(d.^2+4.*d.*y+4.*( ...
  y.^2+z.^2)).^(-1).*(d.^2+4.*d.*y+4.*(x.^2+y.^2+z.^2)).^(-1/2).*(d.^2+4.* ...
  d.*y+4.*(x.^2+y.^2+2.*z.^2)));

end

