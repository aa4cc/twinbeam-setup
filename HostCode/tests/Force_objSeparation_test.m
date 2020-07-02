addpath('..\optim\')
addpath('..\forceModel\')
%%
N_obj = 30;
d_max = 150;

K_reg = 150e-14;
K_rep = 10e-7;

% x = zeros(N_obj, 1);
% y = zeros(N_obj, 1);
x = 500*(2*rand(N_obj,1)-1);
y = 500*(2*rand(N_obj,1)-1);
z = 130*ones(N_obj, 1);
F_des = zeros(3,N_obj);
F_des(3,:) = 3.3400e-11;
F_scale = 1e12;

th = linspace(0, 2*pi, 50);
r = 25;

figure(1)
clf
plotElArray
axis equal tight
hold on

disp('Select position of the manipulated object')
[x_man, y_man] = ginput(1);
plot(x_man + r*cos(th), y_man + r*sin(th), 'r', 'LineWidth', 2)
plot(x_man + d_max*cos(th), y_man + d_max*sin(th), 'k--', 'LineWidth', 1)
x(1) = x_man;
y(1) = y_man;

disp('Select the goal position')
[x_g, y_g] = ginput(1);
plot(x_g, y_g, 'rx', 'MarkerSize', 20, 'LineWidth', 2)

F_des(1:2, 1) = p_reg([x_man;y_man], [x_g;y_g], K_reg);
% plot_force([x_man;y_man;0], F_des(1:2, 1), 'r-', 'LineWidth', 2);

disp('Select positions of the neighboring objects')
for i=2:N_obj
%     [x_n, y_n] = ginput(1);
%     x(i) = x_n;
%     y(i) = y_n;
    plot(x(i) + r*cos(th), y(i) + r*sin(th), 'k--', 'LineWidth', 2)
end

p = [x';y';z'];
%%
F_rep = repulsive_forces(p(:), d_max, K_rep);

F_des = F_des(:) + F_rep;
plot_force(p, F_des, 'r-', 'LineWidth', 2);
%%
for i=1:N_obj
    F_des((1:2) + 3*(i-1)) = clipTheForce(F_des((1:2) + 3*(i-1)), p(1 + 3*(i-1)), p(2 + 3*(i-1)), maxF);
end
plot_force(p, F_des, 'm-', 'LineWidth', 2);

hold off

p = p(:);
F_des = F_des(:);
%%
el_num = 56;

phases = 2*pi*rand(el_num,1);
ur = cos(phases); ui = sin(phases);
%
% F_des = giveMeForce(p, ur, ui);
% %
% profile on
tic
% for i = 1:100
[phases_opt, penalty, F_dev] = findPhasesFoGivenF(p*1e-6, F_des, 200, 100);
% [phases_opt, penalty] = findPhasesFoGivenF_restart(p, F_des);
% end
toc
% profile viewer

penalty
%

I = ~isnan(F_des);
p_red = reshape(p(I), 3, sum(I(:))/3);
F_dev = reshape(F_dev, 3, sum(I(:))/3);
hold on
plot_force(p_red, F_dev, 'b', 'LineWidth', 1)    
hold off

%%
function F_des = p_reg(p, p_des, K)
%P_REG  Position regulator
%
% Syntax:  F_des = p_reg(p, F_des)
%
% Inputs:
%      p         - 3D positions of manipulated object [microns]
%                  (i.e. [p1_x;p1_y;p1_z])
%      p_des     - required 3D positions of manipulated object [microns]
%                  (i.e. [p1_x;p1_y;p1_z])
%
% Outputs:
%     F_des      - force needed to steer the object toward the required
%     position

F_des = K*(p_des(1:2) - p(1:2));

end

function plot_force(p, F, varargin)
    F_scale = 1e12;
    N_obj = numel(p)/3;
    
    for i=1:N_obj
        plot([0 F_scale*F(1 + 3*(i-1))]+p(1 + 3*(i-1)), [0 F_scale*F(2 + 3*(i-1))]+p(2 + 3*(i-1)), varargin{:});
    end
end

function F_rep = repulsive_forces(p, d_max, K)
%REPULSIVE_FORCES computes forces repulsing neighboring object from the
%manipulated object
%
% Syntax:  F_rep = repulsive_forces(p, d_max, K)
%
% Inputs:
%      p         - 3D positions of manipulated object [microns]
%                  (i.e. [p1_x;p1_y;p1_z;p2_x;p2_y;...])
%      p_des     - required 3D positions of manipulated object [microns]
%                  (i.e. [p1_x;p1_y;p1_z])
%
% Outputs:
%     F_des      - force needed to steer the object toward the required
%     position

N_obj = numel(p)/3;
F_rep = zeros(3*N_obj, 1);

for i=2:N_obj
    d = [p(3*(i-1)+1)-p(1); p(3*(i-1)+2)-p(2)];
    d_nom = norm(d);
    if d_nom < d_max
        F_rep((1:2) + 3*(i-1)) = K*d/norm(d)^3;
    else
        F_rep((1:3) + 3*(i-1)) = nan;
    end
end

end
