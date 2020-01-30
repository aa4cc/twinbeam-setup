#include <iostream>
#include <thread>
#include <chrono>
#include <stdint.h>
#include "genControl.h"

using namespace std;
using namespace std::chrono_literals;

int main( int argc, char** argv ) {
    int fp;
    uint16_t phases[56]  = {0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270, 0, 90, 180, 270};
    uint8_t enables[56]  = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    uint8_t disables[56] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	
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

    // Send some phases    
    generator_setPhases(fp, phases, enables);

    // Close the connection
    generator_close(fp);
}