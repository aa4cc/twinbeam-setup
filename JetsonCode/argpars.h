/**
 * @author  Martin Gurtner
 */
#ifndef ARGPARS_H
#define ARGPARS_H

#include "cxxopts.hpp"
#include "AppData.h"

class Options{
public:

    enum class ImageType {
        RAW_G,
        RAW_R,
        BACKPROP_G,
        BACKPROP_R,
    };

    static bool verbose;
    static bool debug;
    static bool show;
    static bool show_markers;
    static ImageType displayImageType;
    static bool savevideo;
    static bool mousekill;
    static bool rtprio;
    static bool beadsearch;
    
    static cxxopts::ParseResult parse(AppData& appData, int argc, char* argv[]);
};

#endif