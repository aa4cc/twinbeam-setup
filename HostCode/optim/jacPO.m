function J = jacPO(ur, ui, Psi, Omega)

    J = zeros(3,56);
    
%     dur = diag(ur);
%     dui = diag(ui);
    
    for i=1:3
        Psi_sym = Psi{i} + Psi{i}';
%         J(i,:) = -dui*(Psi{i} + Psi{i}')*ur + dur*(Psi{i} + Psi{i}')*ui - dui*Omega{i}*ui - dur*Omega{i}*ur;
        J(i,:) = -ui.*(Psi_sym*ur) + ur.*(Psi_sym*ui) - ui.*(Omega{i}*ui) - ur.*(Omega{i}*ur);
    end
end

