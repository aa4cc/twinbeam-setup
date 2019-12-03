#!/usr/bin/env python3


import serial
import time
import numpy
import logging
import itertools
import struct

logger = logging.getLogger(__name__)

def in_range(x, low, high):
    if x < low:
        raise ValueError("Value {} is lower than allowed minimum {}".format(x, low));
    if x > high:
        raise ValueError("Value {} is higher than allowed maximum {}".format(x, high));
    return x

def binary(x, lenght):
    out = [int(i) for i in bin(x)[2:]]
    if len(out) > lenght:
        print("Unable to convert",x,"to binary of len", lenght,"- result is instead",len(out),"bits long.")
        print("The problematic result was",out)
        print("Quiting.")
        sys.exit()
    while len(out) < lenght:
        out = [0] + out
    return out

class Generator(object):
    OPEN_CODE = bytes((255,255,240))
    CLOSE_CODE = bytes((255,255,241))

    OPEN_CODE_FREQ  = bytes((255,255,242))
    CLOSE_CODE_FREQ = bytes((255,255,243))

    OSC_FREQUENCY = 50e6

    """docstring for Generator"""
    def __init__(self, port= None):
        self.port = port
        self.osc_frequency = self.OSC_FREQUENCY

    def set(self, phases, dutys):
        if len(phases) != len(dutys):
            raise ValueError("Phases and dutys must have same length")

        if len(phases) < 64:
            phases = phases + [0 for x in range(64-len(phases))]
            dutys = dutys + [0 for x in range(64-len(dutys))]

        phases.reverse()
        dutys.reverse()

        phases_converted = map(lambda x: in_range(x, 0, 360), phases)
        dutys_converted = map(lambda x: int(in_range(x, 0, 1)*360), dutys)

        stream = itertools.chain.from_iterable(zip(phases_converted, dutys_converted))         

        bit_stream = "".join("{0:09b}".format(x) for x in stream)

        # slit by 8 bits and convert to bytes
        assert(len(bit_stream)%8 == 0)
        byte_stream = [int(bit_stream[8*i:8*(i+1)], 2) for i in range(len(bit_stream)//8)]

        byte_stream.reverse()

        # compute checksum
        chcksum = sum(bytes(byte_stream))

        packet = self.OPEN_CODE + bytes(byte_stream) + self.CLOSE_CODE + struct.pack(">H", chcksum)

        if self.port:
            self.port.write(packet)
        return packet

    def find_PLL(self, frequency):
        if(frequency == 300e3):
            return 3*36, 5, 10 # Known value for 300kHz
        elif(frequency == 40e3):
            return 36, 5, 25 # Known value for 40kHz
        raise ValueError("Settings is only known for 300 kHz and 40 kHz");

    def set_frequency(self, frequency):
        M, N, C = self.find_PLL(frequency)
        self.setup_pll(M, N, C)

    def setup_pll(self, M, N, C):
        chain = []

        chain.extend([0,0]) #reserved bits
        chain.extend([0,0,1,1,0,0,0]) #loop filter
        chain.extend([0,0,0,0,0,0]) #VCO post + reserved
        chain.extend([0,0,1]) #charge pump settings

        #have not found documentation on how to set the previous settings properly,
        #theese seem to work.

        data = [N,M,C,C,C,C,C]

        #M,N,C
        for i in range(7):
            this = data[i]
            if(this==1):
                chain.append(1)
                this = 0
            else:
                chain.append(0)
            high = this//2
            low = high
            oddDiv = 0
            if this%2: 
                high = high + 1 #as per altpll reconfig manual
                oddDiv = 1
            chain.extend(binary(high, 8))
            chain.append(oddDiv) #set odd division
            chain.extend(binary(low, 8))

        scanchain = ""
        for x in chain:
            scanchain += str(x)

        print("Generated schanchain of", len(chain), "bits.")
        print("Scanchain is as follows:")
        print(scanchain)

        #shift right

        scanchain = scanchain[:-1]
        scanchain = '0' + scanchain

        print('Shifted into:')
        print(scanchain)

        scanchainString = scanchain

        scanchain = [scanchain[i:i+8] for i in range(0, len(scanchain), 8)]

        intsArray = []

        for x in scanchain:
            z = x#[::-1]
            y =  int(z, 2)
            intsArray.append(y)

        intsArray.reverse() # Reversed order for transmission

        print("Split it into:", scanchain)
        print("Actually sending bytes   : ",intsArray)

        if self.port:
            self.port.write(self.OPEN_CODE_FREQ + bytes(intsArray) + self.CLOSE_CODE_FREQ)
