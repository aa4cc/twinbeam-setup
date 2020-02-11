function J = jac(p, ur, ui)

    J = zeros(3,56);
    
    dur = diag(ur);
    dui = diag(ui);
    
    [ Psi, Omega ] = modelMatricesPsiOmega( p );

    for i=1:3
        J(i,:) = -dui*(Psi{i} + Psi{i}')*ur + dur*(Psi{i} + Psi{i}')*ui - dui*Omega{i}*ui - dur*Omega{i}*ur;
    end
end

