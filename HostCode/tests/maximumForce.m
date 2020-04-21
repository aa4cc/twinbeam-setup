addpath('../optim/')
addpath('../forceModel/')
addpath('../utils/')
%% Load config vars
el_num = 56;
err_max = 1e-5;
z = 130e-6;
step=25e-6;
file_prefix = 'step25_8way_';
x_coords = -750e-6:step:750e-6;
y_coords = -750e-6:step:750e-6;
    
%% Run the simulation
tic
parfor thread_index=1:8
    force_candidates = 1e-11:1e-11:1e-9;
    candidates_len = numel(force_candidates);
    
    x_size = numel(x_coords);
    y_size = numel(y_coords);
    total_size = x_size * y_size;
    
    % output
    max_forces_array = zeros(y_size, x_size);
    for y=1:y_size
        row_time_start = tic;
        for x=1:x_size
            p = [x_coords(x); y_coords(y); z];
            max_force = force_candidates(1);
            index = 1;
            element_index = (y-1)*x_size + x;
            failed_count = 0;
            for force = force_candidates
                if thread_index == 1 % right
                    F_des = [force; 0; 3.3400e-11];
                elseif thread_index == 2 % left
                    F_des = [-force; 0; 3.3400e-11];
                elseif thread_index == 3 % top
                    F_des = [0; force; 3.3400e-11];
                elseif thread_index == 4 % bottom
                    F_des = [0; -force; 3.3400e-11];
                elseif thread_index == 5 % top-left
                    F_des = [-force; force; 3.3400e-11];
                elseif thread_index == 6 % top-right
                    F_des = [force; force; 3.3400e-11];
                elseif thread_index == 7 % bottom-left
                    F_des = [-force; -force; 3.3400e-11];
                elseif thread_index == 8 % bottom-right
                    F_des = [force; -force; 3.3400e-11];
                else
                    F_des = [0; 0; 0];
                    error('Error - wrong thread index')
                end
                success = 0;
                for i = 1:100
                    if i == 1
                        [phases_opt, penalty] = findPhasesFoGivenF(p, F_des);
                    else
                        [phases_opt, penalty] = findPhasesFoGivenF_restart(p, F_des);
                    end
                    if penalty < err_max
                        max_force = force;
                        success = 1;
                        break;
                    end
                end
                if success == 0 % failed
                    failed_count = failed_count + 1
                    if failed_count >= 3
                        break;
                    end
                else
                    failed_count = 0; % reset counting, we are looking for 3 successive fails
                end
                index = index + 1;
            end
            max_forces_array(y, x) = max_force
        end
        row_time = toc(row_time_start);
        display(sprintf('ID: %d | Progress: %.1f%% | Time left: %.1f min', thread_index, 100*y/y_size, row_time*(y_size-y)/60))
    end

    s = struct;
    if thread_index == 1 % right
        s.max_forces_right = max_forces_array;
        parsave_mat(file_prefix, 'max_forces_right', s);
        parsave_csv(file_prefix, 'max_forces_right', max_forces_array);
    elseif thread_index == 2 % left
        s.max_forces_left = max_forces_array;
        parsave_mat(file_prefix, 'max_forces_left', s);
        parsave_csv(file_prefix, 'max_forces_left', max_forces_array);
    elseif thread_index == 3 % top
        s.max_forces_top = max_forces_array;
        parsave_mat(file_prefix, 'max_forces_top', s);
        parsave_csv(file_prefix, 'max_forces_top', max_forces_array);
    elseif thread_index == 4 % bottom
        s.max_forces_bottom = max_forces_array;
        parsave_mat(file_prefix, 'max_forces_bottom', s);
        parsave_csv(file_prefix, 'max_forces_bottom', max_forces_array);
    elseif thread_index == 5 % top-left
        s.max_forces_top_left = max_forces_array;
        parsave_mat(file_prefix, 'max_forces_top_left', s);
        parsave_csv(file_prefix, 'max_forces_top_left', max_forces_array);
    elseif thread_index == 6 % top-right
        s.max_forces_top_right = max_forces_array;
        parsave_mat(file_prefix, 'max_forces_top_right', s);
        parsave_csv(file_prefix, 'max_forces_top_right', max_forces_array);
    elseif thread_index == 7 % bottom-left
        s.max_forces_bottom_left = max_forces_array;
        parsave_mat(file_prefix, 'max_forces_bottom_left', s);
        parsave_csv(file_prefix, 'max_forces_bottom_left', max_forces_array);
    elseif thread_index == 8 % bottom-right
        s.max_forces_bottom_right = max_forces_array;
        parsave_mat(file_prefix, 'max_forces_bottom_right', s);
        parsave_csv(file_prefix, 'max_forces_bottom_right', max_forces_array);
    end
end
toc

%% Plot the results
% First load .mat files from the simulation!

[X,Y] = meshgrid(x_coords,y_coords);

figure
ax = subplot(3, 3, 1);
surf(X,Y, max_forces_top_left);
pbaspect(ax,[1 1 1])
view(0, 90);
xlim([x_coords(1) x_coords(end)]);
ylim([y_coords(1) y_coords(end)]);
title('Top-left');
colorbar;
shading interp;
hold on;
plotElArray();

ax = subplot(3, 3, 2);
surf(X,Y, max_forces_top);
pbaspect(ax,[2 2 2])
view(0, 90);
xlim([x_coords(1) x_coords(end)]);
ylim([y_coords(1) y_coords(end)]);
title('Top');
colorbar;
shading interp;
hold on;
plotElArray();

ax = subplot(3, 3, 3);
surf(X,Y, max_forces_top_right);
pbaspect(ax,[1 1 1])
view(0, 90);
xlim([x_coords(1) x_coords(end)]);
ylim([y_coords(1) y_coords(end)]);
title('Top-right');
colorbar;
shading interp;
hold on;
plotElArray();

ax = subplot(3, 3, 4);
surf(X,Y, max_forces_left);
pbaspect(ax,[1 1 1])
view(0, 90);
xlim([x_coords(1) x_coords(end)]);
ylim([y_coords(1) y_coords(end)]);
title('Left');
colorbar;
shading interp;
hold on;
plotElArray();

ax = subplot(3, 3, 5);
pbaspect(ax,[1 1 1]);
plotElArray();
colorbar;
xlim([x_coords(1) x_coords(end)]);
ylim([y_coords(1) y_coords(end)]);

ax = subplot(3, 3, 6);
surf(X,Y, max_forces_right);
pbaspect(ax,[1 1 1])
view(0, 90);
xlim([x_coords(1) x_coords(end)]);
ylim([y_coords(1) y_coords(end)]);
title('Right');
colorbar;
shading interp;
hold on;
plotElArray();

ax = subplot(3, 3, 7);
surf(X,Y, max_forces_bottom_left);
pbaspect(ax,[1 1 1])
view(0, 90);
xlim([x_coords(1) x_coords(end)]);
ylim([y_coords(1) y_coords(end)]);
title('Bottom-left');
colorbar;
shading interp;
hold on;
plotElArray();

ax = subplot(3, 3, 8);
surf(X,Y, max_forces_bottom);
pbaspect(ax,[1 1 1])
view(0, 90);
xlim([x_coords(1) x_coords(end)]);
ylim([y_coords(1) y_coords(end)]);
title('Bottom');
colorbar;
shading interp;
hold on;
plotElArray();

ax = subplot(3, 3, 9);
surf(X,Y, max_forces_bottom_right);
pbaspect(ax,[1 1 1])
view(0, 90);
xlim([x_coords(1) x_coords(end)]);
ylim([y_coords(1) y_coords(end)]);
title('Bottom-right');
colorbar;
shading interp;
hold on;
plotElArray();

%%
function parsave_mat(prefix, fname, s)
  save(strcat(prefix, fname, '.mat'), '-struct', 's');
end

function parsave_csv(prefix, fname, max_forces)
  csvwrite(strcat(prefix, fname, '.csv'), max_forces);
end
