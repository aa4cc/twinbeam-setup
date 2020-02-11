tb = twinbeam('147.32.86.177', 30000);
tb.start();
pause(2);
%% Calibrate
tb.calib([1 1 1 1]);
%%
pos_img = tb.positions();
pos_el  = tb.imgCoords2elCoords(double(pos_img));
%
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
figure(1)
subplot(121)
[x, y] = ginput(1);
pos_img_closest = [y, x];

hold on
h_pos = plot(0, 0, 'go');
hold off

for i=1:10000
%     disp(i)
    tic
    pos_img_closest_new  = double(tb.positions_closest(pos_img_closest));
    toc
    
    pos_el_closest  = tb.imgCoords2elCoords(double(pos_img_closest_new));
    
%     fprintf('(%d, %d)\n', pos_img_closest_new(1), pos_img_closest_new(2));

    h_pos.XData = pos_img_closest_new(:,2);
    h_pos.YData = pos_img_closest_new(:,1);
    
    if norm(pos_img_closest - pos_img_closest_new) < 70
        pos_img_closest = pos_img_closest_new;
    end
    
    pause(0.05)
    
% 
%     subplot(122)
%     hold on
%     plot(pos_el_closest(:,1), pos_el_closest(:,2), 'go')
%     hold off
end

%% Tracker
figure(1)
subplot(121)
[x, y] = ginput(1);
tb.tracker_init([y, x]);

hold on
h_pos = plot(0, 0, 'go');
hold off

for i=1:10000
%     disp(i)
    tic
    pos_img_closest_new  = double(tb.tracker_read());
    toc
    
    pos_el_closest  = tb.imgCoords2elCoords(double(pos_img_closest_new));
    
%     fprintf('(%d, %d)\n', pos_img_closest_new(1), pos_img_closest_new(2));

    h_pos.XData = pos_img_closest_new(:,2);
    h_pos.YData = pos_img_closest_new(:,1);
    
    if norm(pos_img_closest - pos_img_closest_new) < 70
        pos_img_closest = pos_img_closest_new;
    end
    
    pause(0.05)
    
% 
%     subplot(122)
%     hold on
%     plot(pos_el_closest(:,1), pos_el_closest(:,2), 'go')
%     hold off
end
%%
tb.stop();
