tb = twinbeam('147.32.86.177',30000);
%%
tb.stop()
tb.settings(1200, 1200, 1352, 596, 5e6, 100, 1, 3100, 2400, 30, 100);
pause(1)
tb.start()
%% Choose the bead to track
N_bp = 2;
img = tb.get();
pos_px = tb.positions();
imshow(img')
hold on
plot(pos_px(:,1), pos_px(:,2), 'gx')
hold off

pos_px = cell(N_bp, 1);
pos_el = cell(N_bp, 1);
for i=1:N_bp
    [x_refi, y_refi] = ginput(1);
    pos_px{i} = [x_refi, y_refi];
end
%%
hold on
while(true)
    new_positions = double(tb.positions());
    for i=1:N_bp
        dist = (new_positions(:,1)-pos_px{i}(end,1)).^2 + (new_positions(:,2)-pos_px{i}(end,2)).^2;
        [~, I_min] = min(dist);
        if (dist(I_min) < 50^2)
            pos_px{i}(end+1,:) = new_positions(I_min,:);
        else
            error('Bead lost!')
            break;
        end
        plot(pos_px{i}(end,1), pos_px{i}(end,2), 'rx')
    end
%     plot(new_positions(:,1), new_positions(:,2), 'r.')
end
hold off
%%
delete(tb)