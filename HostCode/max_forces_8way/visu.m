% Load the mat-files
load('step25_8way_max_forces_bottom_right.mat')
load('step25_8way_max_forces_left.mat')
load('step25_8way_max_forces_top.mat')
load('step25_8way_max_forces_top_left.mat')
load('step25_8way_max_forces_top_right.mat')
load('step25_8way_max_forces_right.mat')
load('step25_8way_max_forces_bottom.mat')
load('step25_8way_max_forces_bottom_left.mat')
% Set the grid
xd = -750:25:750;
yd = -750:25:750;

%% Store the maximum forces in a structure
maxF.br = max_forces_bottom_right*sqrt(2);
maxF.b  = max_forces_bottom;
maxF.bl = max_forces_bottom_left*sqrt(2);
maxF.l  = max_forces_left;
maxF.r = max_forces_right;
maxF.tr = max_forces_top_right*sqrt(2);
maxF.t = max_forces_top;
maxF.tl = max_forces_top_left*sqrt(2);

maxF.xd = xd;
maxF.yd = yd;

save ../maxF maxF

%% plot the max forces
figure(1)
clf

% Top row
subplot(3,3,1)
imagesc(xd, yd, max_forces_bottom_right)
title('Bottom-right direction')
plotElArray()
xlabel('x [um]')
ylabel('y [um]')
axis equal
axis([xd(1) xd(end) yd(1) yd(end)])

subplot(3,3,2)
imagesc(xd, yd, max_forces_bottom)
title('Bottom direction')
plotElArray()
xlabel('x [um]')
ylabel('y [um]')
axis equal
axis([xd(1) xd(end) yd(1) yd(end)])

subplot(3,3,3)
imagesc(xd, yd, max_forces_bottom_left)
title('Bottom-left direction')
plotElArray()
xlabel('x [um]')
ylabel('y [um]')
axis equal
axis([xd(1) xd(end) yd(1) yd(end)])

% Middle row
subplot(3,3,4)
imagesc(xd, yd, max_forces_right)
title('Rigth direction')
plotElArray()
xlabel('x [um]')
ylabel('y [um]')
axis equal
axis([xd(1) xd(end) yd(1) yd(end)])


subplot(3,3,6)
imagesc(xd, yd, max_forces_left)
title('Left direction')
plotElArray()
xlabel('x [um]')
ylabel('y [um]')
axis equal
axis([xd(1) xd(end) yd(1) yd(end)])

% Bottom row
subplot(3,3,7)
imagesc(xd, yd, max_forces_top_right)
title('top-right direction')
plotElArray()
xlabel('x [um]')
ylabel('y [um]')
axis equal
axis([xd(1) xd(end) yd(1) yd(end)])

subplot(3,3,8)
imagesc(xd, yd, max_forces_top)
title('Top direction')
plotElArray()
xlabel('x [um]')
ylabel('y [um]')
axis equal
axis([xd(1) xd(end) yd(1) yd(end)])

subplot(3,3,9)
imagesc(xd, yd, max_forces_top_left)
title('Top-left direction')
plotElArray()
xlabel('x [um]')
ylabel('y [um]')
axis equal
axis([xd(1) xd(end) yd(1) yd(end)])