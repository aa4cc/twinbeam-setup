function [phases, penalty] = findPhasesFoGivenF(p, F_des)
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
%      p    - 3D positions of the objects stored sequentially
%             (i.e. [p1_x;p1_y;p1_z;p2_x;p2_y;...])
%     F_des - the 3D forces to be developed stored seqeuntially 
%             (i.e. [F1_x;F1_y;F1_z;F2_x;F2_y;...])
%
% Outputs:
%     phases - the found phase shifts resulting in the development of the
%              required forces
%    penalty - the norm of the deviation between the generated and required
%              force.
        
el_num = 56;
phases = randn(el_num,1);
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
mu = 10e-2; % initial damping

% fprintf('|Iteration|Penalty |Damping |LSCount|\n');
% fprintf('|---------|--------|--------|-------|\n');
for i = uint32(1:50) % number of iterations
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
%             fprintf('|%9u|%8.2e|%8.2e|%7u|\n',i,penalty,mu,num_search);
            J = jacPO_2xSym(c, s, Psi_2xSym, Omega)/F_norm;
            mu = mu/2;
            break;
        end
        mu = mu*2;
        num_search = num_search + 1;
        if (num_search > 10)
%             fprintf('LMsolve failed to converge on iteration %u\n',i);
            penalty = nan;
            phases = nan(el_num,1);
            break;
        end
    end
    if (penalty < 1e-5 || isnan(penalty)) % penalty tolerance
        break;
    end
end


end

