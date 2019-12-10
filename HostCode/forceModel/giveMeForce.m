function F = giveMeForce( p, ur, ui )

[ Psi, Omega ] = modelMatricesPsiOmega( p );

F = [ur'*Psi{1}*ur + ui'*Psi{1}*ui + ur'*Omega{1}*ui;
     ur'*Psi{2}*ur + ui'*Psi{2}*ui + ur'*Omega{2}*ui;
     ur'*Psi{3}*ur + ui'*Psi{3}*ui + ur'*Omega{3}*ui;];

end

