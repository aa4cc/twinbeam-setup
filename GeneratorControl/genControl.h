#ifndef GENERATOR_CONTROL_H
#define GENERATOR_CONTROL_H

#include "stdio.h"
#include "string.h"
#include "stdlib.h"
#include "stdint.h"

//#include "rtwtypes.h"

#ifdef __cplusplus
extern "C" {
#endif

//! default duty cycle (180 == 50 %) !//
#define DEFAULT_DUTY		180		

/**
 * @brief Open the serial port for communication with the generator.
 * 
 * @param port The serial port to be opened
 * @param baud Baudrate
 * @return the file descriptor
 */
int generator_open(const char *port, int baud);

/**
 * @brief Sets the frequency of the generated signals to 300 kHz.
 * 
 * @param fp file descriptor to the serial port.
 */
void generator_setFreq300kHz(int fp);

/**
 * @brief Close the serial port for communication with the generator.
 * 
 * @param fp The serial port to be opened.
 * @return int 0 if close properly, -1 otherwise.
 */
int generator_close(int fp);

/**
 * @brief Sends data packet to the generator.
 * 
 * @param fp file descriptor to the serial port.
 * @param packet an array of 'len' bytes to be sent to the generator.
 * @param len number of bytes in the packet.
 */
void generator_sendPacket(int fp, const void *packet, int len);

/**
 * @brief Sets phase shifts of the 56 generated signals.
 * 
 * @param fp file descriptor to the serial port.
 * @param phases an array of uint16 phase shifts of the generated signals. The phase shifts are in degrees.
 * @param enables an array of '0's or '1's indicating whether the corresponding channels will have non-zero duty cycle.
 */
void generator_setPhases(int fp, const uint16_t* phases, uint8_t *enables);

#ifdef __cplusplus
}
#endif

#endif