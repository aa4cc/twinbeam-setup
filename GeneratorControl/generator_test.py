#!/usr/bin/env python3
from generator import Generator
from time import sleep
import serial

import curses
 
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

for i in range(0, curses.COLORS):
        curses.init_pair(i + 1, i, -1)


port = serial.Serial('/dev/ttyUSB1', 115200, parity= serial.PARITY_EVEN) #pozn.: paritu prvni verze generatoru v FPGA nekontroluje, paritni bit jen precte a zahodi (aktualni k 8.4.2018)
# port_ver = serial.Serial('/dev/ttyUSB2', 115200, parity= serial.PARITY_EVEN, timeout=1) #pozn.: paritu prvni verze generatoru v FPGA nekontroluje, paritni bit jen precte a zahodi (aktualni k 8.4.2018)

g = Generator(port);

f_phase_4seq = [lambda i : 180 if i%2==0 else 0,
                  lambda i : 180 if i%3==0 else 0,
                lambda i : 180 if i%14==0 else 0,
                lambda i : 180 if i%28==0 else 0,
                lambda i : 180 if i%56==0 else 0];
                
def print_4seq(screen, phases, duty_cycles):
    x_offset = 5;
    y_offset = 1;
    pos_x = [];
    pos_y = [];
    # Top sector
    pos_x.extend([x_offset+3+5*i for i in range(14)]);
    pos_y.extend([y_offset     for i in range(14)]);
    # Right sector
    pos_x.extend([x_offset+3+5*14 for i in range(14)]);
    pos_y.extend([y_offset+i+1    for i in range(14)]);
    # Bottom sector
    pos_x.extend([x_offset+5*(14-i)-2 for i in range(14)]);
    pos_y.extend([y_offset+14+1 for i in range(14)]);
    # Left sector
    pos_x.extend([x_offset for i in range(14)]);
    pos_y.extend([y_offset+14-i for i in range(14)]);

    screen.addstr(0,0,'Phases', curses.color_pair(10))
    for i in range(14*4):
        if phases[i]==180:
            color =  curses.color_pair(11)
        else:
            color = curses.color_pair(7);
        screen.addstr(pos_y[i], pos_x[i], '{:^3}'.format(phases[i]), color);

    screen.addstr(19,0,'Press q to quit', curses.A_STANDOUT)

    screen.refresh()


g.set_frequency(300e3);
# g.set_frequency(40e3);

try:
    i = 0;
    j = 0;
    while True:
        screen.clear()
        phases 		= [f_phase_4seq[j](i+k) for k in range(4*14)];
        duty_cycles = [0.5]*4*14;

        print_4seq(screen, phases, duty_cycles);

        packet = g.set(phases, duty_cycles);
        packet_str = " ".join(("{:02x}".format(x) for x in packet))


        # packet_ver = port_ver.read_until(b"\xff\xff\xf1")

        # if packet_ver:
        # 	packet_ver_str = " ".join(("{:02x}".format(x) for x in packet_ver))

        # 	packet_cmp = [x==y for x,y in zip(packet, packet_ver)]			
        # 	packet_cmp_str = "  ".join(("T" if x else "F" for x in packet_cmp))

        # 	screen.addstr(20,0,packet_str, curses.color_pair(10))  
        # 	screen.addstr(25,0,packet_ver_str, curses.color_pair(10))  
        # 	screen.addstr(30,0,packet_cmp_str, curses.color_pair(10))  

        # 	if all(packet_cmp):
        # 		screen.addstr(35,0,"OK", curses.color_pair(11))  
        # 	else:
        # 		screen.addstr(35,0,"Not OK", curses.color_pair(10))  


        screen.refresh()


        char = screen.getch()
        if char == ord('q'):
            break
        elif char == curses.KEY_RIGHT:
            i -= 1;
        elif char == curses.KEY_LEFT:
            i += 1;
        elif char == curses.KEY_UP:
            j = (j+1)%len(f_phase_4seq);
        elif char == curses.KEY_DOWN:
            j = (j-1)%len(f_phase_4seq);

        # input("Press Enter to continue...")
        # sleep(0.02)

finally:
    # shut down cleanly
    curses.nocbreak(); screen.keypad(0); curses.echo()
    curses.endwin()