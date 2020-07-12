/**
 * @author  Martin Gurtner
 */
#ifndef ARGPARS_H
#define ARGPARS_H

#include "cxxopts.hpp"
#include "AppData.h"

class Options{
public:
    static bool verbose;
    static bool debug;
    static bool show;
    static bool show_fullscreen;
    static bool show_markers;
    static bool show_labels;
    static bool keyboard;
    static ImageType displayImageType;
    static bool savevideo;
    static bool mousekill;
    static bool rtprio;
    static bool beadsearch_R;
    static bool beadsearch_G;
    static uint16_t tcp_port;
    
    static cxxopts::ParseResult parse(AppData& appData, int argc, char* argv[]);
};

#endif