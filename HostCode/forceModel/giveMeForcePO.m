function F = giveMeForcePO( ur, ui, Psi, Omega )

N_comps = numel(Omega);
F = zeros(N_comps, 1);

for i=1:N_comps
    F(i) = ur'*Psi{i}*ur + ui'*Psi{i}*ui + ur'*Omega{i}*ui;
end

end

