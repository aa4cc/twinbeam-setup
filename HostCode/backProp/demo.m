%% Raw
figure(1)
clf

subplot(121)
imshow(img)
title('R channel')

%% Backprop
img = double(img);
lambda = 625e-9;

n = 1.45;
dx = 1.55e-6;

for z = 2800
    subplot(122)
    Erz = rsBackPropMud_fun(img, z*1e-6, dx, n, lambda);
    imagesc(Erz);
    axis equal;
    colormap gray;
    title(sprintf('z = %d um', z))
    
%     pause(0.5)
    waitforbuttonpress
end