%% Necessary Constants
% Offsets [height, width]
% Values of height and width currently need to be even
offsets = [0 0];
% Cut dimensions [height, width]
% Values of height and width currently need to be even
dimensions = [3000 4000];
% illumination source wave length
lambda = 520e-9;
% Refraction index
n = 1.45;
% pixel size
dx = 1.85e-6;
% Back-prop. distances
z = 2000e-6;
% Load stored raw data from the camera
% Currently only set up with the original picture due to lack of time.
raw_data = load('zeroHeightExpFrame');
raw_data = raw_data.frame;

%In matlab I run the algo through a for loop so that I can easily measure
%its average computation time.

tic
counter = 0;
time = 0;

for i = 1:1
    [Erz, ConvArr, Red, RedMax] = Process(raw_data, lambda, n, dx, z(1), int32(offsets), int32(dimensions));
    counter = counter +1;
    %Displaying final picture
    %%{
    figure(1)
    imagesc(Erz);
    [I, J] = find(ConvArr > 30);
    hold on
    plot(J+30, I+30, 'b*')
    hold off
    axis equal;
    colormap gray;
    pause(1e-3);
    
    figure(3)
    imagesc(Red);
    [I, J] = find(RedMax);
    hold on
    plot(J+30, I+30, 'b*')
    hold off
    axis equal;
    colormap gray;
    pause(1e-3);
    %}
end
disp(toc/counter)