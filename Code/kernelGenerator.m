m = 50;
kernel = zeros(m*m,1);


for i = 1:m*m
    temp = sqrt((mod(i,m) - m/2)^2 + (i/m - m/2)^2);
    if temp <= 20
        kernel(i) = -0.45;
    elseif (temp > 22 && temp <= 25)
        kernel(i) = 1.1;
    end
end

kernel2D = zeros(m,m);
for i = 1:m
    for j = 1:m
        kernel2D(i,j) = kernel(j + (i-1) * m);
    end
end



figure(1)
imagesc(kernel2D)
colormap gray;