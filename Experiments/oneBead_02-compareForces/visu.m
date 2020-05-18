load('simout_data.mat')
% load('simout_data_2.mat')

%%
N_obj = 1;

time = simout.time;

ref_pos      = simout.signals(1).values(:, 1:2*N_obj);
meas_pos     = simout.signals(1).values(:,  (1:3*N_obj) + 2*N_obj);
F_des_trim   = simout.signals(1).values(:,  (1:3*N_obj) + (2+3)*N_obj);
F_dev        = simout.signals(1).values(:,  (1:3*N_obj) + (2+3+3)*N_obj);

penalty      = simout.signals(1).values(:,  1 + (2+3+3+3)*N_obj);

LocForceTrim      = simout.signals(1).values(:,  2 + (2+3+3+3)*N_obj);
LocForceTrim_8dir = simout.signals(1).values(:,  3 + (2+3+3+3)*N_obj);
LocForceTrim_max  = simout.signals(1).values(:,  4 + (2+3+3+3)*N_obj);
%% Ref pos
plot(time, ref_pos)
%% Meas pos
plot(time, meas_pos)

%%
recordVideo = true;
if recordVideo
    v = VideoWriter('video.avi','Motion JPEG AVI');
    open(v);
end

th = linspace(0, 2*pi, 50);
x_circ = 50*cos(th);
y_circ = 50*sin(th);

figure(25)
clf
plotElArray();
hold on
for k=1:N_obj
    f_obj_meas(k) = plot(x_circ, y_circ, 'LineWidth', 3);
    f_obj_ref(k) = plot(1, 1, 'rx', 'LineWidth', 5, 'MarkerSize', 20);
    f_Fdev(k) = plot(1,1,'b-', 'LineWidth', 3);
    f_Fdes(k) = plot(1,1,'g-', 'LineWidth', 2);
    f_forceSet(k) = plot(1, 1);
end
hold off
axis equal
axis([-850 850 -850 850])
f_title = title(sprintf('Time: %5.2f', 0));
h_locTrim = text(-810, -780, 'LocBasedTrim: 0', 'BackgroundColor', [1 1 1]);
h_locTrim_8dir = text(-110, -780, '8-dir: 0', 'BackgroundColor', [1 1 1]);
h_locTrim_maxF = text(300, -780, 'max forces: 0', 'BackgroundColor', [1 1 1]);
%%
Fscale = 1e12;

for I=1:1:numel(time)    
    
    for i=1:N_obj
        f_obj_ref(k).XData = ref_pos(I, 1 + 2*(i-1));
        f_obj_ref(k).YData = ref_pos(I, 2*i);
        
        f_obj_meas(k).XData = meas_pos(I, 1 + 3*(i-1)) + x_circ;
        f_obj_meas(k).YData = meas_pos(I, 2 + 3*(i-1)) + y_circ;
        
        F_dev_I = [F_dev(I, 1 + 3*(i-1)); F_dev(I, 2 + 3*(i-1))];
        f_Fdev(k).XData = meas_pos(I, 1 + 3*(i-1)) + Fscale*[0 F_dev_I(1)];
        f_Fdev(k).YData = meas_pos(I, 2 + 3*(i-1)) + Fscale*[0 F_dev_I(2)];
        
        F_des_I = [F_des_trim(I, 1 + 3*(i-1)); F_des_trim(I, 2 + 3*(i-1))];  
        f_Fdes(k).XData = meas_pos(I, 1 + 3*(i-1)) + Fscale*[0 F_des_I(1)];
        f_Fdes(k).YData = meas_pos(I, 2 + 3*(i-1)) + Fscale*[0 F_des_I(2)];   
        
        plot_ForceSet(meas_pos(I, 1 + 3*(i-1)), meas_pos(I, 2 + 3*(i-1)), maxF, 'linear', Fscale, true, f_forceSet(k));
        
        if LocForceTrim(I)
            h_locTrim.String = 'LocBasedTrim: 1';
        else
            h_locTrim.String = 'LocBasedTrim: 0';
        end
        
        if LocForceTrim_8dir(I)
            h_locTrim_8dir.String = '8-dir: 1';
        else
            h_locTrim_8dir.String = '8-dir: 0';
        end
        
        if LocForceTrim_max(I)
            h_locTrim_maxF.String = 'max forces: 1';
        else
            h_locTrim_maxF.String = 'max forces: 0';
        end
    end
    
    f_title.String = sprintf('Time: %5.2f', time(I));    
    
    if recordVideo
        frame = getframe(gcf);
        writeVideo(v,frame);
    end
    
    drawnow
end

if recordVideo
    close(v);
end
