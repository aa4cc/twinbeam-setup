classdef twinbeam < handle
    %TWINBEAM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        connection
        width
        height
        coordinates
        H_calib = []; % calibration matrix
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
                    write(obj.connection, [uint8('r'), 0]);
                case 'backpropagated_r'
                    write(obj.connection, [uint8('r'), 1]);
                case 'raw_g'
                    write(obj.connection, [uint8('r'), 2]);
                case 'raw_r'
                    write(obj.connection, [uint8('r'), 3]);
                otherwise
                    error('This image type is not supported.')
            end
            
            image = typecast(read(obj.connection, obj.height*obj.width), 'uint8');
            image2D = reshape(image, obj.width, obj.height)'; % OpenCV stores images row-wise hence the received image has to be transposed to comply with Matlab notation
            if nargout == 0
                figure(1)
                imshow(image2D);
                axis equal;
                colormap gray;
            else
                ret = image2D;
            end
        end
        
        function obj = settings(obj, width, height, offset_x, offset_y, offset_r2g_x, offset_r2g_y, exposure, analog_gain, digital_gain, red_dist, green_dist, fps, image_threshold_g, image_threshold_r)
            if nargin == 1
                prompt = {'Width:','Height:', ...
                    'Offset X:', 'Offset Y:', ...
                    'R2G Offset X:', 'R2G Offset Y:', ...
                    'Exposure time [ns]:', 'Analog gain:', 'Digital gain:', ...
                    'Red distance [um]:', 'Green distance [um]:', ...
                    'Frames per second: ', 'Image threshold G:', 'Image threshold R:'};
                dlgtitle = 'New settings';
                dims = [1 40];
                definput = {'1024','1024','1440','592','480','0','5000000', '200', '2', '3100', '2400', '30', '90', '140'};
                data = inputdlg(prompt,dlgtitle,dims,definput);
                data = str2double(data);
                width = data(1);
                height = data(2);
            else
                data = [width; height; offset_x; offset_y; offset_r2g_x; offset_r2g_y; exposure; analog_gain; digital_gain; red_dist; green_dist; fps; image_threshold_g; image_threshold_r];
            end

            obj.width = width;
            obj.height = height;
            
            message = [uint8('o')];
            for i=1:numel(data)
                message = [message twinbeam.uint2binarray(uint32(data(i)))];
            end
            write(obj.connection, message);
        end

        function stop(obj)
            write(obj.connection, uint8('q'));
        end
        
        function img_subs(obj, port)
            write(obj.connection, [uint8('b'), typecast(uint16(port), 'uint8')]);
        end
        
        function img_unsubs(obj)
            write(obj.connection, uint8('b'));
        end
        
        function coords_subs(obj, port)
            write(obj.connection, [uint8('c'), typecast(uint16(port), 'uint8')]);
        end
        
        function coords_unsubs(obj)
            write(obj.connection, uint8('c'));
        end
        
        function green = positions(obj)
            write(obj.connection, uint8('g'));
            num_of_coords = typecast(read(obj.connection, 4), 'uint32');
            if num_of_coords == 0
                green = [];
                disp("No green coordinates found");
            else
                coords = typecast(read(obj.connection, num_of_coords*4), 'uint16');
                green = reshape(coords, 2, num_of_coords)';
                % The application running on Jetson indexes the images by
                % (col_id, row_id) whereas Matlab uses (row_id, col_id).
                % Thus we have to switch the received the indexes so that
                % they comply with the Matlab notation.
                green = [green(:,2) green(:,1)];
            end
        end
        
        function tracker_init(obj, inArg)
            % positions are in format [x1 y1; x2 y2; ...]
            
            if numel(inArg)==1 
                % The argument specifies the number of objects to be
                % tracked and the user is supposed to select them in the
                % captured image
                img = obj.get();
                fig = figure(25);
                imshow(img);
                
                [x, y] = ginput(inArg);
                close(fig);
                positions = [y, x];
            else
                positions = inArg;
            end
            
            % The application running on Jetson indexes the images by
            % (col_id, row_id) whereas Matlab uses (row_id, col_id).
            % Thus we have to switch the received the indexes so that
            % they comply with the Matlab notation.
            positions = [positions(:,2)'; positions(:,1)'];
%             message = [uint8('t'), uint8('i'), typecast(uint32(numel(positions)/2),'uint8'), typecast(uint16(positions(:)'),'uint8')];
            message = [uint8('t'), uint8('i'), typecast(uint16(numel(positions)/2),'uint8'), typecast(uint16(numel(positions)/2),'uint8'), typecast(uint16(positions(:)'),'uint8'), typecast(uint16(positions(:)'),'uint8')];
            write(obj.connection, message);
        end
        
        function green = tracker_read(obj)
            % positions are in format [x1 y1; x2 y2; ...]
            
            % The application running on Jetson indexes the images by
            % (col_id, row_id) whereas Matlab uses (row_id, col_id).
            % Thus we have to switch the received the indexes so that
            % they comply with the Matlab notation.
            message = [uint8('t'), uint8('r')];
            write(obj.connection, message);
            
            num_of_coords = typecast(read(obj.connection, 4), 'uint32');
            if num_of_coords == 0
                green = [];
                disp("No green coordinates found");
            else
                coords = typecast(read(obj.connection, num_of_coords*4), 'uint16');
                green = reshape(coords, 2, num_of_coords)';
                % The application running on Jetson indexes the images by
                % (col_id, row_id) whereas Matlab uses (row_id, col_id).
                % Thus we have to switch the received the indexes so that
                % they complie with the Matlab notation.
                green = [green(:,2) green(:,1)];
            end
        end        
        
        function delete(obj)
            write(obj.connection, uint8('d'));
            pause(0.2);
        end
        
        function H = calib(obj, offsets, img_type)
            if nargin < 2
                offsets = [5 5 5 5];
            end
            
            if nargin < 3
                img_type = 'backpropagated';
            end
            % Capture an image of the electrode arrat
            img = obj.get(img_type);
            img = im2single(img);
            [w, h] = size(img);
            
            % Blur the image
            sigma = 4;
            im_filt = imgaussfilt(img, sigma);
            
            % Set the distance from the edge of the limits where the
            % centers of the electrodes will be searched for

            % Find peaks in intensity along the lines crossing the
            % electrodes with $offset pixels from the edge of the image            
            l_l = im_filt(:, offsets(1));
            l_r = im_filt(:, h-offsets(2));
            l_t = im_filt(offsets(3), :)';
            l_b = im_filt(w-offsets(4), :)';

            [pks_l, lcs_l] = findpeaks(l_l);
            [pks_r, lcs_r] = findpeaks(l_r);
            [pks_t, lcs_t] = findpeaks(l_t);
            [pks_b, lcs_b] = findpeaks(l_b);
            
            % Get rid of peaks that are too close to the bondary of the image
            I1 = lcs_l > 60 & lcs_l < (obj.width-60);
            pks_l = pks_l(I1); lcs_l = lcs_l(I1);
            
            I2 = lcs_r > 60 & lcs_r < (obj.width-60);
            pks_r = pks_r(I2); lcs_r = lcs_r(I2);
            
            I3 = lcs_t > 60 & lcs_t < (obj.height-60);
            pks_t = pks_t(I3); lcs_t = lcs_t(I3);
            
            I4 = lcs_b > 60 & lcs_b < (obj.height-60);
            pks_b = pks_b(I4); lcs_b = lcs_b(I4);
            
    
            % Find the maximum values in the peaks ...
%             maxval = max([pks_l;pks_r;pks_t;pks_b]);

            % .. and throw away all peaks smaller then $src*$maxval
            [~, Itmp] = sort(pks_l, 'descend');
            I1 = pks_l>= pks_l(Itmp(14));
            pks_l = pks_l(I1); lcs_l = lcs_l(I1);
            
            [~, Itmp] = sort(pks_r, 'descend');
            I2 = pks_r>= pks_r(Itmp(14));
            pks_r = pks_r(I2); lcs_r = lcs_r(I2);
            
            [~, Itmp] = sort(pks_t, 'descend');
            I3 = pks_t>= pks_t(Itmp(14));
            pks_t = pks_t(I3); lcs_t = lcs_t(I3);
            
            [~, Itmp] = sort(pks_b, 'descend');
            I4 = pks_b>= pks_b(Itmp(14));
            pks_b = pks_b(I4); lcs_b = lcs_b(I4);
            
%             [pks_l, lcs_l] = twinbeam.addMissingPoints(pks_l(I1), lcs_l(I1));
%             
%             I2 = pks_r> sc*maxval;
%             [~, Itmp] = sort(pks_l, 'descend');
%             I1 = pks_l>= pks_l(Itmp(14));
%             [pks_r, lcs_r] = twinbeam.addMissingPoints(pks_r(I2), lcs_r(I2));
%             I3 = pks_t> sc*maxval;
%             [pks_t, lcs_t] = twinbeam.addMissingPoints(pks_t(I3), lcs_t(I3));
%             I4 = pks_b> sc*maxval;
%             [pks_b, lcs_b] = twinbeam.addMissingPoints(pks_b(I4), lcs_b(I4));
                        
            
            % Show the captured electrode array and found centers of the
            % electrodes
            figure(1)
            imagesc(im_filt)
            colormap gray
            axis equal
            axis tight
            
            pts_l = [lcs_l, offsets(1)*ones(size(lcs_l))];
            pts_r = [lcs_r, (h-offsets(2))*ones(size(lcs_r))];
            pts_t = [offsets(3)*ones(size(lcs_t)),        lcs_t];
            pts_b = [(w-offsets(4))*ones(size(lcs_b)),    lcs_b];

            hold on
            plot([offsets(1)  offsets(1)], [offsets(1) w-offsets(1)], 'b--')
            plot([h-offsets(2)  h-offsets(2)], [offsets(2) w-offsets(2)], 'b--')
            plot([h-offsets(3)  offsets(3)], [offsets(3) offsets(3)], 'b--')
            plot([h-offsets(4) offsets(4)], [w-offsets(4) w-offsets(4)], 'b--')

            plot(pts_l(:,2), pts_l(:,1), '*')
            plot(pts_r(:,2), pts_r(:,1), '*')
            plot(pts_t(:,2), pts_t(:,1), '*')
            plot(pts_b(:,2), pts_b(:,1), '*')
            hold off            
            
            % Show the intensities across the electrodes and plot the found
            % centers
            figure(2)
            plot([l_l l_r l_t l_b])
            hold on
            plot(lcs_l, pks_l, '*')
            plot(lcs_r, pks_r, '*')
            plot(lcs_t, pks_t, '*')
            plot(lcs_b, pks_b, '*')
            hold off
            
            figure(1)
                hold on
            for i = 1:14
                plot([pts_l(i,2) pts_r(i,2)], [pts_l(i,1) pts_r(i,1)], 'w--')
                plot([pts_t(i,2) pts_b(i,2)], [pts_t(i,1) pts_b(i,1)], 'w--')
            end

            % Find the ends of the electrodes by the intersection of the
            % found centers of the electrodes
            el_points = [];
            tmp_i = [1:7 6:-1:1];
            % TOP - Q1
            for i = 1:13
                l1 = [pts_t(i,:);pts_b(i,:)];
                l2 = [pts_l(tmp_i(i),:);pts_r(tmp_i(i),:)];
                [px, py] = twinbeam.lineIntersection(l1, l2);
                el_points = [el_points; px, py];
                plot(py, px, 'ro')
%                 pause(.2)
            end

            % RIGHT - Q2
            for i = 1:13
                l1 = [pts_l(i,:);pts_r(i,:)];
                l2 = [pts_t(15-tmp_i(i),:);pts_b(15-tmp_i(i),:)];
                [px, py] = twinbeam.lineIntersection(l1, l2);
                el_points = [el_points; px, py];
                plot(py, px, 'ro')
%                 pause(.2)
            end

            % BOTTOM - Q3
            for i = 1:13
                l1 = [pts_t(15-i,:);pts_b(15-i,:)];
                l2 = [pts_l(15-tmp_i(i),:);pts_r(15-tmp_i(i),:)];
                [px, py] = twinbeam.lineIntersection(l1, l2);
                el_points = [el_points; px, py];
                plot(py, px, 'ro')
%                 pause(.2)
            end

            % LEFT - Q4
            for i = 1:13
                l1 = [pts_l(15-i,:);pts_r(15-i,:)];
                l2 = [pts_t(tmp_i(i),:);pts_b(tmp_i(i),:)];
                [px, py] = twinbeam.lineIntersection(l1, l2);
                el_points = [el_points; px, py];
                plot(py, px, 'ro')
%                 pause(.2)
            end
            
            % Generate an array of the real positions of the tips of the
            % electrodes
            q1_x = [0:100:(13-1)*100]'-650;
            q1_y = -[0:100:(7-1)*100, 500:-100:0]'+650;
            el_points_real = [];

            el_min_len = 150;
            %Q1
            el_points_real = [el_points_real; q1_x q1_y q1_x q1_y(1)*ones(numel(q1_x),1)+el_min_len];
            %Q2
            el_points_real = [el_points_real; q1_y -q1_x  q1_y(1)*ones(numel(q1_y),1)+el_min_len -q1_x];
            %Q3
            el_points_real = [el_points_real; -q1_x -q1_y -q1_x -q1_y(1)*ones(numel(q1_x),1)-el_min_len];
            %Q4
            el_points_real = [el_points_real; -q1_y q1_x -q1_y(1)*ones(numel(q1_y),1)-el_min_len q1_x];
            
%             figure(4)
%             subplot(121)
%             plot(el_points(:,2), el_points(:,1), 'x');
%             ax = gca;
%             ax.YDir = 'reverse';
%             grid on
%             axis equal
% 
%             figure(4)
%             subplot(122)
%             cla
%             ax = gca;
%             grid on
%             axis equal
%             xlim([-1 1]*(650+el_min_len))
%             ylim([-1 1]*(650+el_min_len))
%             hold on
%             for i=1:size(el_points_real,1)
%                 plot(el_points_real(i,[1 3]), el_points_real(i,[2 4]), 'k-', 'LineWidth', 8)
%                 plot(el_points_real(i,1), el_points_real(i,2), 'rx', 'LineWidth', 8)
%                 pause(.2)
%             end
%             hold off
            % Compute the calibration matrix

            zs1 = el_points(:,1:2)';
            zs2 = el_points_real(:,1:2)';
            obj.H_calib = vgg_H_from_x_lin(zs1,zs2);

            zs = obj.H_calib\[zs2; ones(1,size(zs2,2)) ];
            zs_x = zs(1,:)./zs(3,:);
            zs_y = zs(2,:)./zs(3,:);

            figure(1)
            hold on
            plot(zs_y', zs_x', 'gx')
            hold off
            
            %
            figure(3)
            imagesc(img);
            colormap gray
            
            % Generate position of the points bounding each electrode
            el_pos = twinbeam.genElecetrodePositions();
            
            hold on
            for i = 1:56
                el_pos_i = squeeze(el_pos(i,:,:));
                zs = obj.H_calib\[el_pos_i'; ones(1,4) ];
                zs_x = zs(1,:)./zs(3,:);
                zs_y = zs(2,:)./zs(3,:);
                
                plot(zs_y, zs_x, 'w-')
%                 pause(.1)
            end
            hold off
            axis equal tight
            
            H = obj.H_calib;
        end
        
        function el_pos = imgCoords2elCoords(obj, imgCoords)
            assert(~isempty(obj.H_calib), 'You must do the calibration first (call calib() method)!');
            
            el_pos = zeros(size(imgCoords));
            
            el_pos_tmp = obj.H_calib*[imgCoords'; ones(1, size(imgCoords,1))];
            el_pos(:, 1) = el_pos_tmp(1,:)'./el_pos_tmp(3,:)';
            el_pos(:, 2) = el_pos_tmp(2,:)'./el_pos_tmp(3,:)';
        end
    end
    
    methods(Static)
        function binarray = uint2binarray(data)
            binarray = [ ...
                uint8(bitand(bitshift(data, 0), 255)), ...
                uint8(bitand(bitshift(data, -8), 255)),  ...
                uint8(bitand(bitshift(data, -16), 255)),  ...
                uint8(bitand(bitshift(data, -24), 255))];
        end
        
        function [pks, lcs] = addMissingPoints(pks, lcs)
            dx = median(diff(lcs));
            for i=1:numel(lcs)-1
                if (lcs(i+1) - lcs(i)) > 1.5*dx
                    lcs = [lcs(1:i);
                        lcs(i)/2 + lcs(i+1)/2 ;
                        lcs(i+1:end)];
                    
                    pks = [pks(1:i);
                        pks(i)/2 + pks(i+1)/2 ;
                        pks(i+1:end)];
                end
            end
        end
        
        function [px, py] = lineIntersection(l1, l2)
            % l1 = [x1 y1;x2 y2];
            % l2 = [x3 y3;x4 y4];
            x1 = l1(1,1);
            x2 = l1(2,1);
            y1 = l1(1,2);
            y2 = l1(2,2);
            
            x3 = l2(1,1);
            x4 = l2(2,1);
            y3 = l2(1,2);
            y4 = l2(2,2);
            
            det = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4);
            px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4))/det;
            py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4))/det;
        end
        
        function el_pos = genElecetrodePositions()
            NQ_el = 14;
            el_pos = zeros(4*NQ_el, 4, 2);
                        
            q1_x = (0:100:(NQ_el-1)*100) - (NQ_el-1)/2*100;
            q1_y = -[0:100:(NQ_el/2-1)*100 (NQ_el/2 - 2)*100:-100:-100] + (NQ_el-1)/2*100 - 25;

            el_x = [-25 -25 25 25];
            el_y = [900 0 0 900];

            k = 1;
            for th = 0:-pi/2:-2*pi
                for i=1:numel(q1_x)
                    x_tmp = el_x + q1_x(i);
                    y_tmp = min(el_y + q1_y(i), 900);
                    el_pos(k,:,1) = cos(th)*x_tmp - sin(th)*y_tmp;
                    el_pos(k,:,2) = sin(th)*x_tmp + cos(th)*y_tmp;
                    k = k + 1;
                end
            end
        end
    end
end


% connection = tcpclient('147.32.86.177', 4862, 'ConnectTimeout', 10);
% %%
% % message = single(1);
% message = single([1 2]);
% % message = single([1 2 3]);
% %%
% write(connection, message);
