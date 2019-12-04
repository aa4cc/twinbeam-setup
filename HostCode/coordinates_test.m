tb = twinbeam('147.32.86.177', 30000);
tb.start();
pause(2);
%% Calibrate
tb.calib();
%%
pos_img = tb.positions();
pos_el  = tb.imgCoords2elCoords(double(pos_img));
%%
electrodes = twinbeam.genElecetrodePositions();

figure(1)
clf

subplot(121)
tb.get()
hold on
plot(pos_img(:,2), pos_img(:,1), 'rx')
hold off

subplot(122)
hold on
for i = 1:56
    plot(squeeze(electrodes(i,:,1)), squeeze(electrodes(i,:,2)), 'k-')
end
plot(pos_el(:,1), pos_el(:,2), 'rx')
hold off
axis equal tight
%%
tb.stop();
