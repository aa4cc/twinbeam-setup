function [phases, penalty] = findPhasesFoGivenF_restart(p, F_des)

el_num = 56;
F_norm = norm(F_des);
F_des = F_des/F_norm;
[ Psi, Omega ] = modelMatricesPsiOmega( p );
Psi_2xSym = {Psi{1} + Psi{1}', Psi{2} + Psi{2}', Psi{3} + Psi{3}'};

% fprintf('|Iteration|Penalty |Damping |LSCount|\n');
% fprintf('|---------|--------|--------|-------|\n');

iternum = uint32(0);
maxiter = uint32(50);
penalty_min = inf;
phases_best = zeros(el_num,1);
while iternum < maxiter
    phases = randn(el_num,1);
    c = cos(phases);
    s = sin(phases);

    J = jacPO_2xSym(c, s, Psi_2xSym, Omega)/F_norm;
    F_dev = giveMeForcePO(c, s, Psi, Omega)/F_norm;
    penalty = norm(F_des - F_dev);

    % Levenberg-Marquardt solver
    mu = 10e-2; % initial damping
    failedToConverge = false;
    for i = uint32(1:maxiter) % number of iterations
        rhs = J' * (F_des - F_dev);
        JTJ = J' * J;

        num_search = uint32(1);
        while true
            step = (JTJ + mu*eye(el_num)) \ rhs;
            new_phases = phases + step;
            new_c = cos(new_phases);
            new_s = sin(new_phases);
            new_F = giveMeForcePO(new_c, new_s, Psi, Omega)/F_norm;
            new_penalty = norm(F_des - new_F);
            if new_penalty < penalty
                phases = new_phases;
                c = new_c;
                s = new_s;
                F_dev = new_F;
                penalty = new_penalty;
%                 fprintf('|%9u|%8.2e|%8.2e|%7u|\n',i,penalty,mu,num_search);
                J = jacPO_2xSym(c, s, Psi_2xSym, Omega)/F_norm;
                mu = mu/2;
                break;
            end
            mu = mu*2;
            num_search = num_search + 1;
            if (num_search > 5)
                failedToConverge = true;
                break;
            end
        end
        if (penalty < 1e-5 || failedToConverge) % penalty tolerance
            break;
        end
    end
    
    iternum = iternum + i;
    
    if penalty < penalty_min
        penalty_min = penalty;
        phases_best = phases;
    end
    
    if penalty < 1e-5
        break;
    else
%         fprintf('Restarted\n');
    end
end

phases = phases_best;
penalty = penalty_min;

end