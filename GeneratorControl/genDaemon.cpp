#include <iostream>
#include <thread>
#include <chrono>
#include <stdint.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <pthread.h>
#include "genControl.h"

#define PORT 30001
#define BUFSIZE 200

using namespace std;
using namespace std::chrono_literals;

int main( int argc, char** argv ) {
    int fp;
    uint16_t phases[56]  = {0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270};
    uint8_t enables[56]  = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    uint8_t disables[56] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	// Set the RT priority
	struct sched_param schparam;
	schparam.sched_priority = 20;
	
	printf("INFO: display_thread: setting rt priority to %d\n", schparam.sched_priority);

	int s = pthread_setschedparam(pthread_self(), SCHED_FIFO, &schparam);
	if (s != 0) fprintf(stderr, "WARNING: setting the priority of display thread failed.\n");	

    // Open the connection
    cout << "Opening the serial port for communication" << endl;
    fp = generator_open("/dev/ttyUSB0", 115200);
    if (fp == -1) return -1;

    // Disable all channels
    generator_setPhases(fp, phases, disables);

    // Set frequency to 300 kHz
    cout << "Setting frequency to 300 kHz" << endl;
    generator_setFreq300kHz(fp);

    // Wait a few seconds to let the frequency settle
    this_thread::sleep_for(2s);


    int client;
	sockaddr_in sockName;
	sockaddr_in clientInfo; 
	int mainSocket;
	char buf[BUFSIZE];
	socklen_t addrlen;
	
	mainSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if(mainSocket == -1)
		fprintf(stderr, "ERROR: Couldn't create socket!\n");
	sockName.sin_family = AF_INET;
	sockName.sin_port =	htons(PORT);
	// sockName.sin_addr.s_addr = INADDR_ANY;
	sockName.sin_addr.s_addr = inet_addr("147.32.86.177");    

	// Allow reusing the port
	int yes = 1;
	if ( setsockopt(mainSocket, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(int)) == -1 )
	{
		fprintf(stderr, "Setsocket failed!\n");
		return -1;
	}
	
	// Set the time for the rev() to 1 second
	struct timeval tv;
	tv.tv_sec = 1;
	tv.tv_usec = 0;
	if ( setsockopt(mainSocket, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv) == -1 )
	{
		fprintf(stderr, "Setsocket failed!\n");
		return -1;
	}	

	bind(mainSocket, (sockaddr*)&sockName, sizeof(sockName));

	listen(mainSocket, 10000000);
	while(true){		
		addrlen = sizeof(clientInfo);
		client = accept(mainSocket, (sockaddr*)&clientInfo, &addrlen);
		if (client < 1) continue;

		cout << "INFO: Got a connection from " << inet_ntoa((in_addr)clientInfo.sin_addr) << endl;

		while(true){
			int msg_len = recv(client, buf, BUFSIZE - 1, 0);

			// If no message was received within one second, continue
			if (msg_len == -1)	continue;

			// If the connection was closed, break the loop
			if (msg_len == 0)	break;

			printf("DEBUG: Received %d bytes. MessageType: %c \n", msg_len, buf[0]);

            switch(buf[0]) {
                case 'p':
                    generator_setPhases(fp, (uint16_t *)(buf+1), enables);
                    break;
                case 'd':
                    generator_setPhases(fp, phases, disables);
                    break;
            }
		}
		cout << "INFO: Closing the connection from " << inet_ntoa((in_addr)clientInfo.sin_addr) << endl;
		close(client);

		// Disable all channels
		generator_setPhases(fp, phases, disables);
	}
	close(mainSocket);

    // Close the connection
    generator_close(fp);
}