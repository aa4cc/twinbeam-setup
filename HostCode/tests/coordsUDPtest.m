addpath('../')
%%
tb = twinbeam('147.32.86.177', 30000);
tb.tracker_init(1);

u = udp('0.0.0.0', 'LocalPortMode', 'manual', 'LocalPort', 30006, 'InputBufferSize', 8);
fopen(u);
%%
tb.coords_subs(30006);
for i=1:10000
    tic
    data = uint8(fread(u, 8));
%     N_ojb = typecast(data(1:4), 'uint32');
%     x = typecast(data(5:6), 'uint16');
%     y = typecast(data(7:8), 'uint16');
%     fprintf('N: %d - (%d,%d)\n', N_ojb, x, y);
    toc
end
%%
fclose(u);
delete(u);
delete(tb);