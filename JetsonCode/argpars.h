/**
 * @author  Martin Gurtner
 */
#ifndef ARGPARS_H
#define ARGPARS_H

#include "cxxopts.hpp"
#include "Settings.h"

class Options{
public:
    static bool verbose;
    static bool debug;
    static bool show;
    static bool saveimgs;
    static bool saveimgs_bp;
    // static bool savevideos;
    static bool mousekill;

    static cxxopts::ParseResult parse(int argc, char* argv[]);
};

#endif