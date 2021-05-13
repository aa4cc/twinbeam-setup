/**
 * @author  Martin Gurtner
 */
#ifndef PARAMS_H
#define PARAMS_H

#include <string>
#include <vector>
#include <iostream>
#include "Definitions.h"

class Params{
public:

    bool        verbose 		= false;
    bool        debug 		    = false;
    bool        show 			= false;
    bool        show_fullscreen = false;
    bool        show_markers	= false;
    bool        show_labels	    = false;
    bool        keyboard	    = false;
    bool        savevideo 	    = false;
    bool        rtprio		    = false;
    bool        beadsearch_R	= false;
    bool        beadsearch_G	= false;
    bool        cost            = false;
    uint16_t    tcp_port        = DEFAULT_TCP_PORT;
    ImageType   displayImageType = ImageType::MODULUS;

    int	img_width		= 1024;
    int	img_height		= 1024;
    int	img_offset_X	= 1440;
    int	img_offset_Y	= 592;
    int	img_offset_R2G_X = 550;
    int	img_offset_R2G_Y = 20;
    int	cam_exposure	= 5000000;
    int	cam_analoggain	= 2;
    int	cam_digitalgain	= 1;
    int	cam_FPS			= 30;
    int	backprop_z_R	= 3100;
    int	backprop_z_G	= 2400;
    int	improc_thrs_G	= 90;
    int	improc_thrs_R	= 140;
    int	improc_gaussFiltSigma_G	= 5;
    int	improc_gaussFiltSigma_R	= 7;
    int iters0 = 12;
    int iters = 1;
    std::vector<double> rconstr{-2, 0};
    std::vector<double> iconstr{-1, 1};
    double mu = 0.02;
    double z = 2.303e-3;

    // Constructor
    Params() {};

    void print();
    void parseJSONConfigFile(std::string configFileName);
    void parseJSONIStream(std::istream& i);
    std::string getJSONConfigString();
    void parseCmdlineArgs(int argc, char* argv[]);
};

#endif