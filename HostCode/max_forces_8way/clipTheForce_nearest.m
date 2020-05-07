function clippedForce = clipTheForce_nearest(Fd, xt, yt, maxF, giveMeMaxForce)

if nargin < 5
    giveMeMaxForce = false;
end

[~, Ix] = min(abs(maxF.xd - xt));
[~, Iy] = min(abs(maxF.yd - yt));

Fd_angle = atan2(Fd(2), Fd(1));

angles = -pi:pi/4:pi;
Fmax = [maxF.l(Iy, Ix) maxF.bl(Iy, Ix) maxF.b(Iy, Ix) maxF.br(Iy, Ix) maxF.r(Iy, Ix) maxF.tr(Iy, Ix) maxF.t(Iy, Ix) maxF.tl(Iy, Ix)  maxF.l(Iy, Ix)];

[~, I_ang] = min(abs(Fd_angle - angles));

if giveMeMaxForce
    F_scale = 1;
else
    F_scale = min(norm(Fd)/Fmax(I_ang), 1);
end

clippedForce = F_scale*Fmax(I_ang)*[cos(angles(I_ang)); sin(angles(I_ang))];

end