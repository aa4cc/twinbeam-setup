greenBack = imread('greenBackPropagated.png');
redBack = imread('redBackPropagated.png');
red = imread('redNonBackPropagated.png');
green = imread('greenNonBackPropagated.png');



figure(1)
colormap('gray')

ax1 = subplot(221)
imagesc(green)
title(ax1,"Green Channel Non BackPropagated");

ax2 = subplot(222)
imagesc(red)
title(ax2,"Red Channel Non BackPropagated");

ax3 = subplot(223)
imagesc(greenBack)
title(ax3,"Green Channel BackPropagated");

ax4 = subplot(224)
imagesc(redBack)
title(ax4,"Red Channel BackPropagated");