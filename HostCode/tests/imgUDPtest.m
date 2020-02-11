tb = twinbeam('147.32.86.177', 30000);
tb.img_subs(30005);

u = udp('147.32.86.177',30005,'LocalPort', 30005, 'InputBufferSize', 1024*1024*5);
fopen(u);
%%
img = zeros(1024*1024, 1, 'uint8');
for i=1:100
    tic
    for j=1:128
        img( ((j-1)*1024*8+1):j*1024*8 ) = fread(u, 8*1024); 
    end
    toc
    imshow(reshape(img, 1024,1024));
    pause(.001);
end
phases = zeros(56,1);
fwrite(u, [uint8('p'); typecast(uint16(phases),'uint8')])
%%
fclose(u);
delete(u);
delete(tb);