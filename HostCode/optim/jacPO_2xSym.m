function J = jacPO_2xSym(ur, ui, Psi_2xSym, Omega)

    J = zeros(numel(Omega),56);
    for i=1:numel(Omega)
        J(i,:) = -ui.*(Psi_2xSym{i}*ur) + ur.*(Psi_2xSym{i}*ui) - ui.*(Omega{i}*ui) - ur.*(Omega{i}*ur);
    end
end

