% The oblique light source emits the light under 30°
alpha = deg2rad(30);
n_water = 1.33;
n_air   = 1.0;

% The angle of the oblique light source in the water is:
beta = asin( sin(alpha)*(n_air/n_water));

% height = mut_dist/tan(beta) = K*mut_dist
K = 1/tan(beta)