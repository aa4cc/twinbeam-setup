function y=Psiyzf(x,y,z,d)
y=(-1/2).*pi.^(-1).*((-4).*(d.^2+(-4).*d.*y+4.*(y.^2+z.^2)).^(-1)+4.*( ...
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
