#include "genControl.h"
// #include <stdio.h>
// #include <stdlib.h>
// #include <errno.h>

#include <sys/fcntl.h> 
#include <sys/stat.h>
#include <sys/ioctl.h>      
#include <unistd.h>     
#include <stdio.h>
#include <stdlib.h>
#include <asm/ioctls.h>
#include <asm/termbits.h>
#include <math.h>

int generator_open(const char *port, int baud){

	printf("Opening port %s at %dbd\n", port, baud);
	int fp = open(port, O_EXCL | O_RDWR);
	printf("Port has file descriptor %d\n", fp);

	struct termios2 tio;
	int ret;
	if((ret=ioctl(fp, TCGETS2, &tio))==-1){
		fprintf(stderr, "Getting termios2 structure failed. (%d)\n", ret);
		return -1;
	}

    tio.c_cflag |= PARENB;
    tio.c_cflag &= ~PARODD;
    tio.c_cflag &= ~CBAUD;
	tio.c_cflag |= BOTHER;
	tio.c_lflag &= ~ICANON;
	tio.c_lflag &= ~ISIG;
	tio.c_lflag &= ~IEXTEN;
	tio.c_lflag &= ~ECHO;
	tio.c_lflag &= ~ECHOE;
	tio.c_lflag &= ~ECHOK;
	tio.c_lflag &= ~ECHOCTL;
	tio.c_lflag &= ~ECHOKE;
	tio.c_iflag &= ~ICRNL;
	tio.c_iflag &= ~IXON;
	tio.c_oflag &= ~OPOST;
	tio.c_oflag &= ~ONLCR;

	tio.c_cc[VTIME] = 10;
	tio.c_cc[VMIN] = 0;
	tio.c_ispeed = baud;
	tio.c_ospeed = baud;

	if((ret = ioctl(fp, TCSETS2, &tio))==-1){
		fprintf(stderr, "Setting termios2 structure failed. (%d)\n", ret);
		return -1;
	}

	return fp;
}

int generator_close(int fp){
    return close(fp); 
}

void generator_setFreq300kHz(int fp) {
	const uint8_t openCodeFreq[3]  = {255,255,242};
	const uint8_t closeCodeFreq[3] = {255,255,243};
	const uint8_t setFreq300kHz_data[18]  = {2, 5, 10, 20, 40, 80, 160, 64, 129, 2, 5, 108, 216, 16, 56, 32, 0, 6};	

	generator_sendPacket(fp,openCodeFreq,3);
	generator_sendPacket(fp,setFreq300kHz_data,18);
	generator_sendPacket(fp,closeCodeFreq,3);
}

void generator_sendPacket(int fp, const void *packet, int len){
    write(fp, packet, len);
}

void writeOctet(uint16_t *data, uint8_t *dest) {
	dest[0] = data[0]>>1;
	int i;
	for (i = 1; i < 8; i++) {
		dest[i] = (data[i-1]<<(8-i)) | (data[i]>>(i+1));
	}
	dest[8] = data[7];
}

void generator_setPhases(int fp, const uint16_t *phases, uint8_t *enables) {
	const uint8_t openCode[3] = {255,255,240};
	const uint8_t closeCode[3] = {255,255,241};

	uint8_t data[144];
	uint16_t octet[8];

	uint16_t checksum = 0;
	int i,j;
	for (i = 0; i < 16; i++)
	{
		for (j = 0; j < 4; j++) {
			int k = 4*i+j;

			// The generator outputs 64 signals whereas we need to generate only 56 signals.
			// Therefore, we set zero phase shifts and duty cycles for the unused signals.
			// Furthermore, the signal parameters are sent to the generator in the reverse order.
			if (k >= 8) {
				octet[2*j]   = phases[56-(k-8)-1];
				octet[2*j+1] = enables[56-(k-8)-1] * DEFAULT_DUTY;
			} else {
				octet[2*j] = 0;
				octet[2*j+1] = 0;
			}
		}
		writeOctet(octet, data + 9*i);
	}

	for (i = 0; i < 72; i++) {
		uint8_t temp = data[i];
		checksum += temp;
		data[i] = data[143 - i];
		checksum += data[i];
		data[143 - i] = temp;
	}

	uint16_t checksum_big = (checksum << 8) | (checksum >> 8);

	// printf("[");
	// for(int i=0;i<144;i++)
	// 	printf("%d, ", data[i]);
	// printf("\b\b]\n");

	generator_sendPacket(fp,openCode,3);
	generator_sendPacket(fp,data,144);
	generator_sendPacket(fp,closeCode,3);
	generator_sendPacket(fp,&checksum_big,2);
}
