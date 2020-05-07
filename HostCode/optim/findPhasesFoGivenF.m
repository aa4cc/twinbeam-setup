function [phases, penalty, F_dev, numiter] = findPhasesFoGivenF(p, F_des, maxiter, restartAfter, phasesInit)
%FINDPHASESFORGIVENF  Finds phase shifts developing the required forces.
%
% Syntax:  [phases, penalty] = findPhasesFoGivenF(p, F_des)
%
%   The sedimentation force is not taken into account. The physical parameters
%   are:
%       * four-quadrant electrode array with 56 electrodes and 50 micron
%       electrode and inner gap width
%       * 50 micron spherical microparticles in diameter
%       * 16 V and 300 kHz square wave signal applied to the electrodes
%
% Inputs:
%      p         - 3D positions of the objects stored sequentially
%                  (i.e. [p1_x;p1_y;p1_z;p2_x;p2_y;...])
%     F_des      - the 3D forces to be developed stored seqeuntially 
%                  (i.e. [F1_x;F1_y;F1_z;F2_x;F2_y;...])
%   maxiter      - maximum number of iterations
%   restartAfter - the number of iterations after which the sovler
%                  reinitializes the phases
%   phasesInit   - intialization of the phases
%
% Outputs:
%     phases - the found phase shifts resulting in the development of the
%              required forces
%    penalty - the norm of the deviation between the generated and required
%              force.
%      F_dev - the force developed by the found vector of phase shifts
%    numiter - number of iterations needed to reach the returned phases
        

if nargin < 3
    maxiter = uint32(35);
end

if nargin < 4
    restartAfter = maxiter;
end

el_num = 56;

if nargin < 5
    phasesInit = randn(el_num,1);
end

phases = phasesInit;
c = cos(phases);
s = sin(phases);
F_norm = norm(F_des);
F_des = F_des/F_norm;

N_objs = numel(p)/3;
Psi = cell(3*N_objs,1);
Omega = cell(3*N_objs,1);
Psi_2xSym = cell(3*N_objs,1);

% Compute the model matrices for each manipulated object
for i=1:N_objs
    [ Psi_i, Omega_i ] = modelMatricesPsiOmega( p( 3*(i-1) + (1:3) ) );
    for j=1:3
        I = 3*(i-1)+j;
        Psi{I} = Psi_i{j};
        Omega{I} = Omega_i{j};
        Psi_2xSym{I} = Psi_i{j} + Psi_i{j}';
    end
end
J = jacPO_2xSym(c, s, Psi_2xSym, Omega)/F_norm;
F_dev = giveMeForcePO(c, s, Psi, Omega)/F_norm;
penalty = norm(F_des - F_dev);

%% Levenberg-Marquardt solver
mu_init = 25e-2; % initial damping
mu = mu_init; 

% fprintf('|Iteration|Penalty |Damping |LSCount|\n');
% fprintf('|---------|--------|--------|-------|\n');
i = uint32(1);
i_lastRestart = i;

best_penalty = inf;
best_phases = phases;
doesNotDecrease = false;
while i < maxiter
    rhs = F_dev - F_des;
    
    num_search = uint32(1); % stores the number of iterations needed to decrease the penalty
    while true
        % iterate until the penalty is increase (i.e. increase mu until penalty decreases)
        Abar = [J'; sqrt(mu)*eye(3*N_objs)];
        R = qr(Abar, 0);
        R = R(1:3*N_objs, 1:3*N_objs);
%         step = -J'*((R'*R)\rhs);
        tmp = linsolve(R', rhs, struct('LT', true));
        tmp = linsolve(R,tmp, struct('UT', true));
        step = -J'*tmp;

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
%             fprintf('|%9u|%8.2e|%8.2e|%7u|\n',i,penalty,mu,num_search);
            J = jacPO_2xSym(c, s, Psi_2xSym, Omega)/F_norm;
            mu = mu/2;
            
            if penalty < best_penalty
                best_penalty = penalty;
                best_phases = phases;
            end
            break;
        end
        mu = mu*2;
        num_search = num_search + 1;
        if (num_search > 10)
%             fprintf('LMsolve failed to converge on iteration %u\n',i);
            doesNotDecrease = true;
            break;
        end
    end
    i = i + num_search;
    
    if (penalty < 1e-5) % penalty tolerance
        break;
    end
    
    if doesNotDecrease ||(i-i_lastRestart) >= restartAfter
        % Restart the problem
%         fprintf('Restart\n')
        phases = randn(el_num,1);
        c = cos(phases);
        s = sin(phases);
        J = jacPO_2xSym(c, s, Psi_2xSym, Omega)/F_norm;
        
        F_dev = giveMeForcePO(c, s, Psi, Omega)/F_norm;
        penalty = norm(F_des - F_dev);
        mu = mu_init;
        i_lastRestart = i;
    end    
end

phases = best_phases;
penalty = best_penalty;

numiter = i;

F_dev = F_dev*F_norm;

end

