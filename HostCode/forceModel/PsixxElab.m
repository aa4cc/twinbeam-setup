function [ val ] = PsixxElab( i, x, y, z, a, b )

NQ_el = 14;
v_shift = ( (0:100:(NQ_el-1)*100) - (NQ_el-1)/2*100 )*1e-6;
h_shift = ([0:100:(NQ_el/2-1)*100 (NQ_el/2 - 2)*100:-100:-100] - (NQ_el-1)/2*100 + 25 )*1e-6;

val = 0;
    
if i <= NQ_el
    % Q1
    x_el = -y - h_shift(i);
    y_el = +x - v_shift(i);    
    for i=1:numel(a)
        val = val + a(i) * Psiyyf(x_el-(b(i)-1)*25e-6, y_el, z, b(i)*50e-6);
    end
elseif i > NQ_el && i<=2*NQ_el
    % Q2
    x_el = -x - h_shift(i-NQ_el);
    y_el = -y - v_shift(i-NQ_el);    
    for i=1:numel(a)
        val = val + a(i) * Psixxf(x_el-(b(i)-1)*25e-6, y_el, z, b(i)*50e-6);
    end
elseif i > 2*NQ_el && i<=3*NQ_el
    % Q3
    x_el = +y - h_shift(i-2*NQ_el);
    y_el = -x - v_shift(i-2*NQ_el);    
    for i=1:numel(a)
        val = val + a(i) * Psiyyf(x_el-(b(i)-1)*25e-6, y_el, z, b(i)*50e-6);
    end
elseif i > 3*NQ_el && i<=4*NQ_el
    % Q4
    x_el = x - h_shift(i-3*NQ_el);
    y_el = y - v_shift(i-3*NQ_el);
    for i=1:numel(a)
        val = val + a(i) * Psixxf(x_el-(b(i)-1)*25e-6, y_el, z, b(i)*50e-6);
    end
end


end

