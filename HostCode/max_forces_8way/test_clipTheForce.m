load('../maxF.mat')

% Randomly generate a test force
xt = (maxF.xd(end)-maxF.xd(1))*rand(1) + maxF.xd(1);
yt = (maxF.yd(end)-maxF.yd(1))*rand(1) + maxF.yd(1);

% Randomly generate a test force
Ft = 4e-11*( 2*rand(2,1) - 1);

tic
clippedForce = clipTheForce(Ft, xt, yt, maxF);
toc

% Plot the result
figure(25)
clf
% Plot the el. array and the test position
subplot(211)
plotElArray()
hold on
plot(xt, yt, 'rx')
hold off


subplot(212)
plot_ForceSet(xt, yt, maxF)
hold on

% % lines
% F = [Fmax_t;Fmax_tr;Fmax_r;Fmax_br;Fmax_b;Fmax_bl;Fmax_l;Fmax_tl];
% 
% th = pi/2:-pi/4:-5*pi/4;
% 
% line = [1 1 1;-2 0 2];
% R = @(th) [cos(th) -sin(th); sin(th) cos(th)];
% 
% 
% for i = 1:numel(th)
%     line_tmp = line;
%     line_tmp(1,:) = line_tmp(1,:)*F(i);
%     line_tmp = R(th(i))*line_tmp;
%     plot(line_tmp(1,:), line_tmp(2,:), '-', 'Color', 0.5*[1 1 1])
% %     waitforbuttonpress
% end
% axis(1.1*[-Fmax_l Fmax_r -Fmax_b Fmax_t])

plot([0 clippedForce(1)], [0 clippedForce(2)], 'LineWidth', 4);
plot([0 Ft(1)], [0 Ft(2)]);
hold off
grid on
axis equal
