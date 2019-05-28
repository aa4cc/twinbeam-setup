[y, x] = size(Erz);
step = 200;

set = zeros((y/step)*(x/step), step, step);
%figure(1);
%h = imagesc(zeros(step, step, 'double'));

counter_y = 1;
counter_x = 1;

%% Split picture in even sized squares

for i = 1:(y/step)
    counter_x = 1;
    for j = 1:(x/step)
        set((x/step - 1)*(i-1) + j, : , :) = Erz(counter_y : counter_y + step - 1, counter_x : counter_x + step - 1);
        counter_x = counter_x + step;
    end
    counter_y = counter_y + step;
end

%% Save training pictures 

for i = 1:(y/step)
    for j = 1:(x/step)
        image = squeeze(set((x/step - 1)*(i-1) + j, : , :));
        if ((x/step - 1)*(i-1) + j) >= 100
            name = sprintf("A%d", (x/step - 1)*(i-1) + j);
        elseif ((x/step - 1)*(i-1) + j) >= 10
            name = sprintf("A0%d", (x/step - 1)*(i-1) + j);
        else
            name = sprintf("A00%d", (x/step - 1)*(i-1) + j);
        end
        fnameMat = sprintf('%s.mat', name);
        fpathMat = fullfile(fileparts('\\aspace.msad.fel.cvut.cz\users$\koropvik\Dokumenty\Internship\TrainingData\file'), fnameMat);
        save(fpathMat, 'image');
    end
end

%% User now has to input 