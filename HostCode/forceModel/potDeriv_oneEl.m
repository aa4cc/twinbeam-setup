function [ phi, Ex, Ey, Ez, Psixx, Psixy, Psixz, Psiyy, Psiyz, Psizz ] = potDeriv_oneEl( i, x, y, z, a, b )

NQ_el = 14;
v_shift = ( (0:100:(NQ_el-1)*100) - (NQ_el-1)/2*100 )*1e-6;
h_shift = ([0:100:(NQ_el/2-1)*100 (NQ_el/2 - 2)*100:-100:-100] - (NQ_el-1)/2*100 + 25 )*1e-6;
            
phi     = 0;
Ex      = 0;
Ey      = 0;
Ez      = 0;
Psixx   = 0;
Psixy   = 0;
Psixz   = 0;
Psiyy   = 0;
Psiyz   = 0;
Psizz   = 0;
    
if i <= NQ_el       
    % Q1
    x_el = -y - h_shift(i);
    y_el = +x - v_shift(i);    
    for i=1:numel(a)
        [ phi_i, Ex_i, Ey_i, Ez_i, Psixx_i, Psixy_i, Psixz_i, Psiyy_i, Psiyz_i, Psizz_i ] = ...
            potDeriv_baseEl_optim(x_el-(b(i)-1)*25e-6, y_el, z, b(i)*50e-6);
        phi     = phi   + a(i) * phi_i;
        Ex      = Ex    + a(i) * Ey_i;
        Ey      = Ey    - a(i) * Ex_i;
        Ez      = Ez    + a(i) * Ez_i;
        Psixx   = Psixx + a(i) * Psiyy_i;
        Psixy   = Psixy - a(i) * Psixy_i;
        Psixz   = Psixz + a(i) * Psiyz_i;
        Psiyy   = Psiyy + a(i) * Psixx_i;
        Psiyz   = Psiyz - a(i) * Psixz_i;
        Psizz   = Psizz + a(i) * Psizz_i;
    end
elseif i > NQ_el && i<=2*NQ_el
    % Q2
    x_el = -x - h_shift(i-NQ_el);
    y_el = -y - v_shift(i-NQ_el);    
    for i=1:numel(a)
        [ phi_i, Ex_i, Ey_i, Ez_i, Psixx_i, Psixy_i, Psixz_i, Psiyy_i, Psiyz_i, Psizz_i ] = ...
            potDeriv_baseEl_optim(x_el-(b(i)-1)*25e-6, y_el, z, b(i)*50e-6);
        phi     = phi   + a(i) * phi_i;
        Ex      = Ex    - a(i) * Ex_i;
        Ey      = Ey    - a(i) * Ey_i;
        Ez      = Ez    + a(i) * Ez_i;
        Psixx   = Psixx + a(i) * Psixx_i;
        Psixy   = Psixy + a(i) * Psixy_i;
        Psixz   = Psixz - a(i) * Psixz_i;
        Psiyy   = Psiyy + a(i) * Psiyy_i;
        Psiyz   = Psiyz - a(i) * Psiyz_i;
        Psizz   = Psizz + a(i) * Psizz_i;
    end
elseif i > 2*NQ_el && i<=3*NQ_el
    % Q3
    x_el = +y - h_shift(i-2*NQ_el);
    y_el = -x - v_shift(i-2*NQ_el);    
    for i=1:numel(a)
        [ phi_i, Ex_i, Ey_i, Ez_i, Psixx_i, Psixy_i, Psixz_i, Psiyy_i, Psiyz_i, Psizz_i ] = ...
            potDeriv_baseEl_optim(x_el-(b(i)-1)*25e-6, y_el, z, b(i)*50e-6);
        phi     = phi   + a(i) * phi_i;
        Ex      = Ex    - a(i) * Ey_i;
        Ey      = Ey    + a(i) * Ex_i;
        Ez      = Ez    + a(i) * Ez_i;
        Psixx   = Psixx + a(i) * Psiyy_i;
        Psixy   = Psixy - a(i) * Psixy_i;
        Psixz   = Psixz - a(i) * Psiyz_i;
        Psiyy   = Psiyy + a(i) * Psixx_i;
        Psiyz   = Psiyz + a(i) * Psixz_i;
        Psizz   = Psizz + a(i) * Psizz_i;
    end
elseif i > 3*NQ_el && i<=4*NQ_el
    % Q4
    x_el = x - h_shift(i-3*NQ_el);
    y_el = y - v_shift(i-3*NQ_el);
    for i=1:numel(a)
        [ phi_i, Ex_i, Ey_i, Ez_i, Psixx_i, Psixy_i, Psixz_i, Psiyy_i, Psiyz_i, Psizz_i ] = ...
            potDeriv_baseEl_optim(x_el-(b(i)-1)*25e-6, y_el, z, b(i)*50e-6);
        phi     = phi   + a(i) * phi_i;
        Ex      = Ex    + a(i) * Ex_i;
        Ey      = Ey    + a(i) * Ey_i;
        Ez      = Ez    + a(i) * Ez_i;
        Psixx   = Psixx + a(i) * Psixx_i;
        Psixy   = Psixy + a(i) * Psixy_i;
        Psixz   = Psixz + a(i) * Psixz_i;
        Psiyy   = Psiyy + a(i) * Psiyy_i;
        Psiyz   = Psiyz + a(i) * Psiyz_i;
        Psizz   = Psizz + a(i) * Psizz_i;
    end
end


end

