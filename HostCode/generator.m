classdef generator < handle
    %TWINBEAM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        connection
    end
    
    methods
        function obj = generator(ip, port, varargin)
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
                definput = {'147.32.86.177','30001','10'};
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
            
            obj.connection = tcpclient(ip, port, 'ConnectTimeout', timeout);
            read(obj.connection)
        end
        
        function sendPhases(obj, phases)
            assert(numel(phases)==56, 'You have to provide a vector of phases with 56 elements.');            
            assert(all(phases >=0 & phases < 360), 'The phase-shifts must be non-negative and smaller than 360');
            
            write(obj.connection, [uint8('p'); typecast(uint16(phases),'uint8')]);
        end
       
        function delete(obj)
            pause(0.2);
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
