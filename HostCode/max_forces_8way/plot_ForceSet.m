function plot_ForceSet(xt, yt, maxF, method, Fscale, shift, f_line)

if nargin < 4
    method = 'linear';
end

if nargin < 5
    Fscale = 1;
end

if nargin < 6
    shift = false;
end

if nargin < 7
    f_line = [];
end

% Plot the result
Fmax_tl = interp2(maxF.xd, maxF.yd, maxF.tl, xt, yt, method);
Fmax_t =  interp2(maxF.xd, maxF.yd, maxF.t, xt, yt, method);
Fmax_tr = interp2(maxF.xd, maxF.yd, maxF.tr, xt, yt, method);
Fmax_r =  interp2(maxF.xd, maxF.yd, maxF.r, xt, yt, method);
Fmax_l =  interp2(maxF.xd, maxF.yd, maxF.l, xt, yt, method);
Fmax_bl = interp2(maxF.xd, maxF.yd, maxF.bl, xt, yt, method);
Fmax_b =  interp2(maxF.xd, maxF.yd, maxF.b, xt, yt, method);
Fmax_br = interp2(maxF.xd, maxF.yd, maxF.br, xt, yt, method);

points = [0         Fmax_t;
    sqrt(2)/2*Fmax_tr   sqrt(2)/2*Fmax_tr;
    Fmax_r    0;
    sqrt(2)/2*Fmax_br   -sqrt(2)/2*Fmax_br;
    0         -Fmax_b;
    -sqrt(2)/2*Fmax_bl  -sqrt(2)/2*Fmax_bl;
    -Fmax_l    0;
    -sqrt(2)/2*Fmax_tl  sqrt(2)/2*Fmax_tl;
    0         Fmax_t];

points = points * Fscale;

if shift
    points(:,1) = points(:,1) + xt;
    points(:,2) = points(:,2) + yt;
end

if isempty(f_line)
    hold on
    plot(points(:,1), points(:,2), '-')
    hold off
else
    if isa(f_line, 'matlab.graphics.chart.primitive.Line')
        f_line.XData = points(:,1);
        f_line.YData = points(:,2);
    else
        error('Unsupported line type')
    end
end

end

