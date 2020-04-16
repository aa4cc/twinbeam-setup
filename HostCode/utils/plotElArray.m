function plotElArray()
NQ_el = 14;
el_pos = zeros(4*NQ_el, 4, 2);

q1_x = (0:100:(NQ_el-1)*100) - (NQ_el-1)/2*100;
q1_y = -[0:100:(NQ_el/2-1)*100 (NQ_el/2 - 2)*100:-100:-100] + (NQ_el-1)/2*100 - 25;

el_x = [-25 -25 25 25];
el_y = [900 0 0 900];

k = 1;
for th = 0:-pi/2:-2*pi
    for i=1:numel(q1_x)
        x_tmp = el_x + q1_x(i);
        y_tmp = min(el_y + q1_y(i), 900);
        el_pos(k,:,1) = cos(th)*x_tmp - sin(th)*y_tmp;
        el_pos(k,:,2) = sin(th)*x_tmp + cos(th)*y_tmp;
        k = k + 1;
    end
end

hold on
for i = 1:56
    el_pos_i = squeeze(el_pos(i,:,:));
    
    plot(el_pos_i(:,1), el_pos_i(:,2), 'k-')
%     pause(.1)
end
hold off

end

