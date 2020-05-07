function clippedForce = clipTheForce(Fd, xt, yt, maxF, giveMeMaxForce)

if nargin < 5
    giveMeMaxForce = false;
end

% Check the quadrant
if Fd(1) > 0
    % right half-plane
    if Fd(2) > 0
        % top-right quadrant
        
        % Interpolate the max force along the diagonal
        Fmax_tr = interp2(maxF.xd, maxF.yd, maxF.tr, xt, yt, 'linear');
        
        if(Fd(2) > Fd(1))  
            % The first octant
            % Interpolate the max forces
            Fmax_t = interp2(maxF.xd, maxF.yd, maxF.t, xt, yt, 'linear');
        
            [n, b] = computeTheLineParameters(Fmax_t, Fmax_tr, 'tr1');
        else
            % The second octant
            % Interpolate the max forces
            Fmax_r = interp2(maxF.xd, maxF.yd, maxF.r, xt, yt, 'linear');
            
            [n, b] = computeTheLineParameters(Fmax_tr, Fmax_r, 'tr2');
        end        
    else
        % Interpolate the max force along the diagonal
        Fmax_br = interp2(maxF.xd, maxF.yd, maxF.br, xt, yt, 'linear');
        
        % bottom-right quadrant
        if(Fd(1) > -Fd(2))
            % The third octant
            % Interpolate the max forces
            Fmax_r = interp2(maxF.xd, maxF.yd, maxF.r, xt, yt, 'linear');
            
            [n, b] = computeTheLineParameters(Fmax_r, Fmax_br, 'br1');
        else
            % The forth octant
            % Interpolate the max forces
            Fmax_b = interp2(maxF.xd, maxF.yd, maxF.b, xt, yt, 'linear');
            
            [n, b] = computeTheLineParameters(Fmax_br, Fmax_b, 'br2');
        end
    end
else
    % left half-plane
    if Fd(2) > 0
        % top-left quadrant        
        % Interpolate the max force along the diagonal
        Fmax_tl = interp2(maxF.xd, maxF.yd, maxF.tl, xt, yt, 'linear');
        
        if(Fd(2) > -Fd(1)) 
            % The eight octant
            % Interpolate the max forces
            Fmax_t = interp2(maxF.xd, maxF.yd, maxF.t, xt, yt, 'linear');
        
            [n, b] = computeTheLineParameters(Fmax_tl, Fmax_t, 'tl2');
        else
            % The seventh octant
            % Interpolate the max forces
            Fmax_l = interp2(maxF.xd, maxF.yd, maxF.l, xt, yt, 'linear');
            
            [n, b] = computeTheLineParameters(Fmax_l, Fmax_tl, 'tl1');
        end        
    else
        % bottom-left quadrant
        % Interpolate the max force along the diagonal
        Fmax_bl = interp2(maxF.xd, maxF.yd, maxF.bl, xt, yt, 'linear');
        
        % bottom-right quadrant
        if(-Fd(1) > -Fd(2))
            % The sixth octant
            % Interpolate the max forces
            Fmax_l = interp2(maxF.xd, maxF.yd, maxF.l, xt, yt, 'linear');
            
            [n, b] = computeTheLineParameters(Fmax_bl, Fmax_l, 'bl2');
        else
            % The fifth octant
            % Interpolate the max forces
            Fmax_b = interp2(maxF.xd, maxF.yd, maxF.b, xt, yt, 'linear');
            
            [n, b] = computeTheLineParameters(Fmax_b, Fmax_bl, 'bl1');
        end
    end    
end
    
k = b/(n'*Fd(:));

if ~giveMeMaxForce
    k = min(k, 1);
end

clippedForce = Fd * k;

end

function [n, b] = computeTheLineParameters(Fmax1, Fmax2, line_type)
c = 1;
switch lower(line_type)
    % Top-right quadrant
    case 'tr1'
        x1 = 0;
        y1 = Fmax1;
        x2 = sqrt(2)/2*Fmax2;
        y2 = sqrt(2)/2*Fmax2;
    case 'tr2'
        x1 = sqrt(2)/2*Fmax1;
        y1 = sqrt(2)/2*Fmax1;
        x2 = Fmax2;
        y2 = 0;
        % Bottom-right quadrant
    case 'br1'
        x1 = Fmax1;
        y1 = 0;
        x2 = sqrt(2)/2*Fmax2;
        y2 = -sqrt(2)/2*Fmax2;
        c = -1;
    case 'br2'
        x1 = sqrt(2)/2*Fmax1;
        y1 = -sqrt(2)/2*Fmax1;
        x2 = 0;
        y2 = -Fmax2;
        c = -1;
        % Bottom-left quadrant
    case 'bl1'
        x1 = 0;
        y1 = -Fmax1;
        x2 = -sqrt(2)/2*Fmax2;
        y2 = -sqrt(2)/2*Fmax2;
    case 'bl2'
        x1 = -sqrt(2)/2*Fmax1;
        y1 = -sqrt(2)/2*Fmax1;
        x2 = -Fmax2;
        y2 = 0;
        % Top-left quadrant
    case 'tl1'
        x1 = -Fmax1;
        y1 = 0;
        x2 = -sqrt(2)/2*Fmax2;
        y2 = sqrt(2)/2*Fmax2;
    case 'tl2'
        x1 = -sqrt(2)/2*Fmax1;
        y1 = sqrt(2)/2*Fmax1;
        x2 = 0;
        y2 = Fmax2;
    otherwise
        disp('Unknown line type.')
end

a = (y1-y2)/(x1-x2);
n = c*[-a; 1];
b = c*(y1 -a*x1);
end