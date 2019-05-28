quit = 0;
message = NaN;
width = 1024;
height = 1024;
while quit == 0
    disp("Waiting for input\n");
    in = input('','s');
    switch in
        % type connect to throw a dialog window and later connect to jetson
        case "connect"
            client = 0;
            prompt = {'IP Address:','Port:','Timeout:'};
            dlgtitle = 'Connection settings';
            dims = [1 35];
            % here you will most probably have to change the IP address
            definput = {'147.32.86.177','30000','10'};
            answer = inputdlg(prompt,dlgtitle,dims,definput);
            while client == 0
                try
                    client = tcpclient(char(answer(1)), str2double(answer(2)), 'ConnectTimeout', str2double(answer(3)));
                catch
                    disp("Connection not found, trying again.");
                    client = 0;
                end
            end
            read(client)
        % type stop to put jetson to sleep while remaining connected
        case "stop"
            write(client, uint8('q'));
        % type request while the image processing loop is running to receive current picture and visualize it
        case "request"
            write(client, uint8('r'));
            image = typecast(read(client, height*width*4), 'single');
            image2D = convert22D(image, width, height);
            figure(1)
            imagesc(image2D);
            axis equal;
            colormap gray;
        % type quit to disconnect from jetson and quit this process
        case "quit"
            write(client, uint8('d'));
            
            quit = 1;
        % type start to wake up Jetson
        case "start"
            write(client, uint8('s'));
            read(client);
        % type settings to throw a dialog window that lets you select individual desired settings
        case "settings"
            prompt = {'Width:','Height:','Offset X:', 'Offset Y:', 'Exposure time [ns]:', 'Red distance [um]:', 'Green distance [um]:'};
            dlgtitle = 'New settings';
            dims = [1 40];
            definput = {'1024','1024','1500','1000','5000000', '2500', '2500'};
            answer = inputdlg(prompt,dlgtitle,dims,definput);
            message = uint8(char(strcat("o ", answer(1), " ", answer(2), " ", answer(3), " ", answer(4), " ", answer(5), " ", answer(6), " ", answer(7), " ")));
            write(client, message);
        otherwise
            disp("Unknown command!\n");
    end
end


function output = convert22D(image, width, height)
    output = zeros(height, width);
    for i = 1:width
       output(:,i) = image(1,((i-1)*height+1):(i*height));
    end
end