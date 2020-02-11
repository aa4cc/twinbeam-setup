addpath('..\optim\')
addpath('..\forceModel\')
%%
el_num = 56;

x = 500e-6*(2*rand(1)-1);
y = 500e-6*(2*rand(1)-1);
z = 130e-6;

p = [x;y;z];

phases = 2*pi*rand(el_num,1);
ur = cos(phases); ui = sin(phases);
%
% F_des = giveMeForce(p, ur, ui);
F_des = [-10e-11; 10e-11; 3.3400e-11];
% %
% profile on
tic
% for i = 1:100
[phases_opt, penalty] = findPhasesFoGivenF(p, F_des);
% [phases_opt, penalty] = findPhasesFoGivenF_restart(p, F_des);
% end
toc
% profile viewer

penalty

[F_des giveMeForce(p, cos(phases_opt), sin(phases_opt))]
%% Two objects

x1 = 500e-6*(2*rand(1)-1);
y1 = 500e-6*(2*rand(1)-1);
z1 = 130e-6;

x2 = 500e-6*(2*rand(1)-1);
y2 = 500e-6*(2*rand(1)-1);
z2 = 130e-6;

p = [x1;y1;z1;x2;y2;z2];

phases = 2*pi*rand(el_num,1);
ur = cos(phases); ui = sin(phases);
%
% F_des = giveMeForce(p, ur, ui);
F_des = [-10e-11; 10e-11; 3.3400e-11; -10e-11; 10e-11; 3.3400e-11];
% %
% profile on
tic
% for i = 1:100
[phases_opt, penalty] = findPhasesFoGivenF(p, F_des);
% [phases_opt, penalty] = findPhasesFoGivenF_restart(p, F_des);
% end
toc
% profile viewer

penalty

[F_des giveMeForce(p, cos(phases_opt), sin(phases_opt))]

%% For N objects
N = 4;

