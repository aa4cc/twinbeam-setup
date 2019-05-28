%% Necessary Constants
% Offsets [height, width]
% Values of height and width currently need to be even
offsets = [0 0];
% Cut dimensions [height, width]
% Values of height and width currently need to be even
dimensions = [3000 4000];
% illumination source wave length
lambda = 515e-9;
% Refraction index
n = 1.45;
% pixel size
dx = 1.85e-6;
% Back-prop. distances
z = 2500e-6;

vidobj = videoinput('winvideo', 1, 'YUY2_4000x3000');
src = getselectedsource(vidobj);
%%
% Configure the object for manual trigger mode.
triggerconfig(vidobj, 'manual');

%%
% Now that the device is configured for manual triggering, call START.
% This will cause the device to send data back to MATLAB, but will not log
% frames to memory at this point.
start(vidobj)

figure(1)


axis equal
colormap gray;

% Measure the time to acquire 20 frames.
%{

for i = 1:20
    fprintf('Step: %4d\n', i);
    snapshot = getsnapshot(vidobj);
    
    h.CData = Process(snapshot, lambda, n, dx, z(1), int32(offsets), int32(dimensions));
    
    
    pause(1e-3)
end
%}

snapshot = getsnapshot(vidobj);
img = Process(snapshot, lambda, n, dx, z(1), int32(offsets), int32(dimensions));
imagesc(img);
pause(1e-3)
savedImage = img;

%%
% Call the STOP function to stop the device.
stop(vidobj)
close
clear vidobj