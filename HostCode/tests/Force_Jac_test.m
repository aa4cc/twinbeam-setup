addpath('optim\')
addpath('forceModel\')
%%
el_num = 56;

x = 500e-6*(2*rand(1)-1);
y = 500e-6*(2*rand(1)-1);
z = 130e-6;

p = [x;y;z];

phases = 2*pi*rand(el_num,1);
ur = cos(phases); ui = sin(phases);

[Fx, Fy, Fz] = giveMeForce(p, ur, ui)
%%
dth = 1e-5;
J_num = zeros(el_num, 3);
for i=1:el_num
    phases_l = phases - ((1:el_num)'==i)*dth;
    phases_r = phases + ((1:el_num)'==i)*dth;
    [Fxl, Fyl, Fzl] = giveMeForce(p, cos(phases_l), sin(phases_l));
    [Fxr, Fyr, Fzr] = giveMeForce(p, cos(phases_r), sin(phases_r));
    J_num(i,:) = [Fxr - Fxl, Fyr - Fyl, Fzr - Fzl]/2/dth;
end

J = jac(p, ur, ui);

% [J-J_num]
(J-J_num)./J