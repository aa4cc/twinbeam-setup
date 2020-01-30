function F = giveMeForce( p, ur, ui )

N_objs = numel(p)/3;
F = zeros(3*N_objs, 1);

for i=1:N_objs
    [ Psi, Omega ] = modelMatricesPsiOmega( p( (i-1)*3 + (1:3) ) );

    F((i-1)*3 + (1:3)) = [ur'*Psi{1}*ur + ui'*Psi{1}*ui + ur'*Omega{1}*ui;
                          ur'*Psi{2}*ur + ui'*Psi{2}*ui + ur'*Omega{2}*ui;
                          ur'*Psi{3}*ur + ui'*Psi{3}*ui + ur'*Omega{3}*ui;];
end

end

