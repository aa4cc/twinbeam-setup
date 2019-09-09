#include "Definitions.h"
#include "Settings.h"
#include "Input.h"
#include <iostream>

void changeSettings(char* buf){
	int tmpSettings; 
	int count = 0;
	int current_index = 2;
	while(count < 7){
		string str = "";
		while(isdigit(buf[current_index])){
			str.append(1u,buf[current_index]);
			current_index++;
		}
		try{
			tmpSettings = atol(str.c_str());
		}
		catch(int e ){
			printf("Number is too large\n");
			tmpSettings = 0;
		}
		if(tmpSettings != 0){
			Settings::set_setting(count, tmpSettings);
			printf("%d\n", Settings::values[count]);
		}
		count++;
		current_index++;
	}
}

void input_thread(){
	printf("input_thread: started\n");

	std::string text;
	sockaddr_in sockName;
	sockaddr_in clientInfo; 
	int mainSocket;
	char buf[BUFSIZE];
	socklen_t addrlen;
	MESSAGE_TYPE response;
	
	mainSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if(mainSocket == -1)
		printf("Couldn't create socket!\n");
	sockName.sin_family = AF_INET;
	sockName.sin_port =	htons(PORT);
	sockName.sin_addr.s_addr = INADDR_ANY;

	int yes = 1;
	if ( setsockopt(mainSocket, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(int)) == -1 )
	{
	    perror("setsockopt");
	}

	bind(mainSocket, (sockaddr*)&sockName, sizeof(sockName));
	listen(mainSocket, 10000000);
	while(!force_exit){
		
		addrlen = sizeof(clientInfo);
		client = accept(mainSocket, (sockaddr*)&clientInfo, &addrlen);
		cout << "Got a connection from " << inet_ntoa((in_addr)clientInfo.sin_addr) << endl;
		if (client != -1)
		 {
			 connected = true;
		 }

		while(connected && !force_exit){
			int msg_len = recv(client, buf, BUFSIZE - 1, 0);

			if (msg_len == -1)
			{
				printf("Error while receiving data\n");
			}

			printf("Received bytes: %d\n", msg_len);

			response = parseMessage(buf);
			switch(response){
				case MSG_WAKEUP:
				{
					sleeping = false;
					initialized = true;
					break;
				}
				case MSG_SLEEP:
				{
					sleeping = true;
					initialized = false;
					break;
				}
				case MSG_SETTINGS:
				{
					if(sleeping == false)
						printf("Can't change settings while the loop is running\n");
					else{
						changeSettings(buf);
						printf("Changed settings\n");
					}
					break;
				}
				case MSG_DISCONNECT:
				{
					connected = false;
					sleeping = true;
					initialized = false;
					break;
				}
				case MSG_REQUEST:
					requested_image = true;
					requested_type = BACKPROPAGATED;
					break;
				case MSG_REQUEST_RAW_G:
					requested_image = true;
					requested_type = RAW_G;
					break;
				case MSG_REQUEST_RAW_R:
					requested_image = true;
					requested_type = RAW_R;
					break;
				case MSG_HELLO:
				{
					send(client, "Hello!",7,0);
					break;
				}	
			} 
		}
		close(client);
	}
	close(mainSocket);

	printf("input_thread: ended\n");
}