classdef twinbeam
    %TWINBEAM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        connection
        width
        height
        coordinates
    end
    
    methods
        function obj = twinbeam(ip, port, varargin)
            %TWINBEAM Construct an instance of this class
            %   Detailed explanation goes here
            
            p = inputParser;
            addRequired(p,'ip',@ischar);
            addRequired(p,'port',@isnumeric);
            addOptional(p,'Timeout',10,@isnumeric);
            %addParameter(p,'units',defaultUnits,@isstring);
            
            if nargin == 0
                prompt = {'IP Address:','Port:','Timeout:'};
                dlgtitle = 'Connection settings';
                dims = [1 35];
                % here you will most probably have to change the IP address
                definput = {'147.32.86.177','30000','10'};
                answer = inputdlg(prompt,dlgtitle,dims,definput);
                ip = char(answer(1));
                port = str2double(answer(2));
                timeout = str2double(answer(3));
            else
                parse(p, ip, port, varargin{:});
                ip = p.Results.ip;
                port = p.Results.port;
                timeout = p.Results.Timeout;
            end
            
            obj.height = 1024;
            obj.width = 1024;
            
            obj.connection = tcpclient(ip, port, 'ConnectTimeout', timeout);
            read(obj.connection)
            obj.start()
        end

        function start(obj)
            write(obj.connection, uint8('s'));
            read(obj.connection);
        end
        
        function ret = get(obj, img_type)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            
            if nargin < 2
                img_type = 'backpropagated';
            end
            
            switch lower(img_type)
                case 'backpropagated'
                    write(obj.connection, uint8('r'));
                case 'raw_g'
                    write(obj.connection, uint8('x'));
                case 'raw_r'
                    write(obj.connection, uint8('y'));
                otherwise
                    error('This image type is not supported.')
            end
            
            
            
            image = typecast(read(obj.connection, obj.height*obj.width*4), 'single');
            image2D = reshape(image, obj.width, obj.height);
            if nargout == 0
                figure(1)
                imagesc(image2D);
                axis equal;
                colormap gray;
            else
                ret = image2D;
            end
        end
        
        function obj = settings(obj, width, height, offset_x, offset_y, exposure, red_dist, green_dist)
            if nargin == 1
                prompt = {'Width:','Height:','Offset X:', 'Offset Y:', 'Exposure time [ns]:', 'Red distance [um]:', 'Green distance [um]:'};
                dlgtitle = 'New settings';
                dims = [1 40];
                definput = {'1024','1024','1500','1000','5000000', '2500', '2400'};
                answer = inputdlg(prompt,dlgtitle,dims,definput);
                width = str2double(answer(1));
                height = str2double(answer(2));
                offset_x = str2double(answer(3));
                offset_y = str2double(answer(4));
                exposure = str2double(answer(5));
                red_dist = str2double(answer(6));
                green_dist = str2double(answer(7));
            end

            obj.width = width;
            obj.height = height;
            message = uint8(char(strcat("o ", num2str(width), " ", num2str(height), " ", num2str(offset_x), " ", num2str(offset_y), " ", num2str(exposure), " ", num2str(red_dist), " ", num2str(green_dist), " ")));
            write(obj.connection, message);
        end

        function stop(obj)
            write(obj.connection, uint8('q'));
        end
        
        function [green, red] = positions(obj)
            write(obj.connection, uint8('g'));
            num_of_coords = typecast(read(obj.connection, 4), 'int32');
            if num_of_coords == 0
                green = 0;
                disp("No green coordinates found");
            else
                indeces = typecast(read(obj.connection, num_of_coords*4), 'int32');
                green = zeros(num_of_coords,2);
                for i = 1:num_of_coords
                    green(i,1) = indeces(i);
                end
                
                disp(green);
            end
            num_of_coords = typecast(read(obj.connection, 4), 'int32')
            if num_of_coords == 0
                red = 0;
                disp("No red coordinates found");
            else
                indeces = typecast(read(obj.connection, num_of_coords*4), 'int32');
                red = zeros(num_of_coords,2);
                for i = 1:num_of_coords
                    red(i,1) = indeces(i)/obj.width;
                    red(i,2) = mod(indeces(i),obj.width);
                end
                disp(red);
            end
        end
        
        function delete(obj)
            write(obj.connection, uint8('d'));
            pause(0.2);
        end
    end
end

