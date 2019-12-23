#!/usr/bin/env python3
from generator import Generator
from packet_receiver import Receiver
from time import sleep
import struct
import serial

import curses
                
def print_4seq(screen, phases, duty_cycles):
    x_offset = 5
    y_offset = 1
    pos_x = []
    pos_y = []
    # Top sector
    pos_x.extend([x_offset+3+5*i for i in range(14)])
    pos_y.extend([y_offset     for i in range(14)])
    # Right sector
    pos_x.extend([x_offset+3+5*14 for i in range(14)])
    pos_y.extend([y_offset+i+1    for i in range(14)])
    # Bottom sector
    pos_x.extend([x_offset+5*(14-i)-2 for i in range(14)])
    pos_y.extend([y_offset+14+1 for i in range(14)])
    # Left sector
    pos_x.extend([x_offset for i in range(14)])
    pos_y.extend([y_offset+14-i for i in range(14)])

    screen.clear()
    screen.addstr(0,0,'Phases', curses.color_pair(10))
    for i in range(14*4):
        if phases[i]==180:
            color =  curses.color_pair(11)
        else:
            color = curses.color_pair(7)
        screen.addstr(pos_y[i], pos_x[i], '{:^3}'.format(phases[i]), color)

    screen.addstr(19,0,'Press q to quit', curses.A_STANDOUT)
    screen.refresh()

def updatePhases(generator, screen, phases, duty_cycles=[0.5]*4*14):
    el2gen_indexes = range(56)
    phases_gen      = [phases[i] for i in el2gen_indexes]
    duty_cycles_gen = [duty_cycles[i] for i in el2gen_indexes]
    generator.set(phases_gen, duty_cycles_gen)
    print_4seq(screen, phases, duty_cycles)

    # screen.addstr(20,0, repr(phases), curses.color_pair(10))
    # screen.addstr(25,0, repr(phases_gen), curses.color_pair(10))
    # screen.refresh()

def parsePhases(gen, screen, data):
    if (len(data)%(2*56)) == 0:
        phases = struct.unpack("56H", data[-2*56:])
        updatePhases(gen, screen, list(phases))
    else:
        screen.addstr(20, 0, "Unparsed", curses.color_pair(10))
        screen.addstr(21, 0, repr(data), curses.color_pair(10))
        screen.refresh()


def main():
    port = serial.Serial('/dev/ttyUSB0', 115200, parity= serial.PARITY_EVEN) #pozn.: paritu prvni verze generatoru v FPGA nekontroluje, paritni bit jen precte a zahodi (aktualni k 8.4.2018)
    # port_ver = serial.Serial('/dev/ttyUSB2', 115200, parity= serial.PARITY_EVEN, timeout=1) #pozn.: paritu prvni verze generatoru v FPGA nekontroluje, paritni bit jen precte a zahodi (aktualni k 8.4.2018)

    # get the curses screen window
    screen = curses.initscr()
    curses.start_color()
    curses.use_default_colors()
    # turn off input echoing
    curses.noecho()
    # respond to keys immediately (don't wait for enter)
    curses.cbreak()
    # map arrow keys to special values
    screen.keypad(True)

    # initialize screen colors
    for i in range(0, curses.COLORS):
            curses.init_pair(i + 1, i, -1)

    f_4seq = [  lambda i, val : val if i%2==0 else 0,
                lambda i, val : val if i%3==0 else 0,
                lambda i, val : val if i%14==0 else 0,
                lambda i, val : val if i%28==0 else 0,
                lambda i, val : val if i%56==0 else 0]    

    gen = Generator(port)
    gen.set_frequency(300e3)
    # gen.set_frequency(40e3)

    rec = Receiver(30001,
            {
                "56H": lambda phases: updatePhases(gen, screen, list(phases)),
            },
            lambda data: parsePhases(gen, screen, data)
        )
    rec.start()

    try:
        i = 0
        j = 0
        while True:
            phases 		= [f_4seq[j](i+k, 180) for k in range(4*14)]
            # duty_cycles	= [f_4seq[j](i+k, 0.5) for k in range(4*14)]

            updatePhases(gen, screen, phases)
            
            # screen.refresh()

            char = screen.getch()
            if char == ord('q'):
                break
            elif char == curses.KEY_RIGHT:
                i -= 1
            elif char == curses.KEY_LEFT:
                i += 1
            elif char == curses.KEY_UP:
                j = (j+1)%len(f_4seq)
            elif char == curses.KEY_DOWN:
                j = (j-1)%len(f_4seq)

            # input("Press Enter to continue...")
            # sleep(0.02)

    finally:
        # shut down cleanly
        gen.set([0]*64, [0.0]*64)
        curses.nocbreak()
        screen.keypad(0)
        curses.echo()
        curses.endwin()


if __name__ == "__main__":
    main()