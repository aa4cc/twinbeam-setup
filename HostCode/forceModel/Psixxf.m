function y=Psixxf(x,y,z,d)
y=(1/2).*pi.^(-1).*z.*(x.^2+z.^2).^(-1).*(8.*x.*y.*((d.^2+(-4).*d.*y+4.*( ...
  x.^2+y.^2+z.^2)).^(-3/2)+(-1).*(d.^2+4.*d.*y+4.*(x.^2+y.^2+z.^2)).^( ...
  -3/2))+d.*((-4).*x.*(d.^2+(-4).*d.*y+4.*(x.^2+y.^2+z.^2)).^(-3/2)+(-4).* ...
  x.*(d.^2+4.*d.*y+4.*(x.^2+y.^2+z.^2)).^(-3/2)))+(-1).*pi.^(-1).*x.*z.*( ...
  x.^2+z.^2).^(-2).*(2.*y.*((-1).*(d.^2+(-4).*d.*y+4.*(x.^2+y.^2+z.^2)).^( ...
  -1/2)+(d.^2+4.*d.*y+4.*(x.^2+y.^2+z.^2)).^(-1/2))+d.*((d.^2+(-4).*d.*y+ ...
  4.*(x.^2+y.^2+z.^2)).^(-1/2)+(d.^2+4.*d.*y+4.*(x.^2+y.^2+z.^2)).^(-1/2)) ...
  );
