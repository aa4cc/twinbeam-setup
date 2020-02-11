function [ phi, Ex, Ey, Ez, Psixx, Psixy, Psixz, Psiyy, Psiyz, Psizz ] = potDeriv_oneEl_ab( i, x, y, z )

NQ_el = 14;
v_shift = ( (0:100:(NQ_el-1)*100) - (NQ_el-1)/2*100 )*1e-6;
h_shift = ([0:100:(NQ_el/2-1)*100 (NQ_el/2 - 2)*100:-100:-100] - (NQ_el-1)/2*100 + 25 )*1e-6;

if i <= NQ_el
    % Q1
    x_el = -y - h_shift(i);
    y_el = +x - v_shift(i);
    
    [ phi_i, Ex_i, Ey_i, Ez_i, Psixx_i, Psixy_i, Psixz_i, Psiyy_i, Psiyz_i, Psizz_i ] = ...
        potDeriv_baseEl_ab_optim(x_el, y_el, z);
    phi     = + phi_i;
    Ex      = + Ey_i;
    Ey      = - Ex_i;
    Ez      = + Ez_i;
    Psixx   = + Psiyy_i;
    Psixy   = - Psixy_i;
    Psixz   = + Psiyz_i;
    Psiyy   = + Psixx_i;
    Psiyz   = - Psixz_i;
    Psizz   = + Psizz_i;
elseif i > NQ_el && i<=2*NQ_el
    % Q2
    x_el = -x - h_shift(i-NQ_el);
    y_el = -y - v_shift(i-NQ_el);
    
    [ phi_i, Ex_i, Ey_i, Ez_i, Psixx_i, Psixy_i, Psixz_i, Psiyy_i, Psiyz_i, Psizz_i ] = ...
        potDeriv_baseEl_ab_optim(x_el, y_el, z);
    phi     = + phi_i;
    Ex      = - Ex_i;
    Ey      = - Ey_i;
    Ez      = + Ez_i;
    Psixx   = + Psixx_i;
    Psixy   = + Psixy_i;
    Psixz   = - Psixz_i;
    Psiyy   = + Psiyy_i;
    Psiyz   = - Psiyz_i;
    Psizz   = + Psizz_i;
elseif i > 2*NQ_el && i<=3*NQ_el
    % Q3
    x_el = +y - h_shift(i-2*NQ_el);
    y_el = -x - v_shift(i-2*NQ_el);
    
    [ phi_i, Ex_i, Ey_i, Ez_i, Psixx_i, Psixy_i, Psixz_i, Psiyy_i, Psiyz_i, Psizz_i ] = ...
        potDeriv_baseEl_ab_optim(x_el, y_el, z);
    phi     = + phi_i;
    Ex      = - Ey_i;
    Ey      = + Ex_i;
    Ez      = + Ez_i;
    Psixx   = + Psiyy_i;
    Psixy   = - Psixy_i;
    Psixz   = - Psiyz_i;
    Psiyy   = + Psixx_i;
    Psiyz   = + Psixz_i;
    Psizz   = + Psizz_i;
elseif i > 3*NQ_el && i<=4*NQ_el
    % Q4
    x_el = x - h_shift(i-3*NQ_el);
    y_el = y - v_shift(i-3*NQ_el);
    
    [ phi_i, Ex_i, Ey_i, Ez_i, Psixx_i, Psixy_i, Psixz_i, Psiyy_i, Psiyz_i, Psizz_i ] = ...
        potDeriv_baseEl_ab_optim(x_el, y_el, z);
    phi     = + phi_i;
    Ex      = + Ex_i;
    Ey      = + Ey_i;
    Ez      = + Ez_i;
    Psixx   = + Psixx_i;
    Psixy   = + Psixy_i;
    Psixz   = + Psixz_i;
    Psiyy   = + Psiyy_i;
    Psiyz   = + Psiyz_i;
    Psizz   = + Psizz_i;
else
    phi     = nan;
    Ex      = nan;
    Ey      = nan;
    Ez      = nan;
    Psixx   = nan;
    Psixy   = nan;
    Psixz   = nan;
    Psiyy   = nan;
    Psiyz   = nan;
    Psizz   = nan;
end


end

