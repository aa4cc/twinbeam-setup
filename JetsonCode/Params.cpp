/**
 * @author  Martin Gurtner
 */
#include <fstream>
#include <stdexcept>
#include "Params.h"
#include "nlohmann/json.hpp"
#include "cxxopts.hpp"

using namespace std;
using json = nlohmann::json;

void Params::print() {
    printf("Parameters: \n");
    
    printf("\t verbose:         %s\n",           verbose ? "true": "false");
    printf("\t debug:           %s\n",             debug ? "true": "false");
    printf("\t show:            %s\n",              show ? "true": "false");
    printf("\t show_fullscreen: %s\n",   show_fullscreen ? "true": "false");
    printf("\t show_markers:    %s\n",      show_markers ? "true": "false");
    printf("\t show_labels:     %s\n",       show_labels ? "true": "false");
    printf("\t keyboard:        %s\n",          keyboard ? "true": "false");
    printf("\t savevideo:       %s\n",         savevideo ? "true": "false");
    printf("\t rtprio:          %s\n",            rtprio ? "true": "false");
    printf("\t beadsearch_R:    %s\n",      beadsearch_R ? "true": "false");
    printf("\t beadsearch_G:    %s\n",      beadsearch_G ? "true": "false");

    printf("\t tcp_port:         %10d\n", tcp_port);
    printf("\t img_width:        %10d\n", img_width);
    printf("\t img_height:       %10d\n", img_height);
    printf("\t img_offset_X:     %10d\n", img_offset_X);
    printf("\t img_offset_Y:     %10d\n", img_offset_Y);
    printf("\t img_offset_R2G_X: %10d\n", img_offset_R2G_X);
    printf("\t img_offset_R2G_Y: %10d\n", img_offset_R2G_Y);
    printf("\t cam_exposure:     %10d\n", cam_exposure);
    printf("\t cam_analoggain:   %10d\n", cam_analoggain);
    printf("\t cam_digitalgain:  %10d\n", cam_digitalgain);
    printf("\t cam_FPS:          %10d\n", cam_FPS);
    printf("\t backprop_z_R:     %10d\n", backprop_z_R);
    printf("\t backprop_z_G:     %10d\n", backprop_z_G);
    printf("\t improc_thrs_G:    %10d\n", improc_thrs_G);
    printf("\t improc_thrs_R:    %10d\n", improc_thrs_R);
}

void Params::parseJSONConfigFile(string fileName) {
    // read a JSON config file
    ifstream i(fileName);

    // If the config file does not exist, throw an exception
    if( !i.is_open() ) throw invalid_argument( "Config file does not exist." );

    parseJSONIStream(i);
}

void Params::parseJSONIStream(istream& i) {

    json j;

    try {
        i >> j;
    } catch (json::exception& ex) {
        cerr << ex.what();
    }

    if (j.contains("verbose")) {
        verbose = j["verbose"].get<bool>();
    }
    if (j.contains("debug")) {
        debug = j["debug"].get<bool>();
    }
    if (j.contains("show")) {
        show = j["show"].get<bool>();
    }
    if (j.contains("show_fullscreen")) {
        show_fullscreen = j["show_fullscreen"].get<bool>();
    }
    if (j.contains("show_markers")) {
        show_markers = j["show_markers"].get<bool>();
    }
    if (j.contains("show_labels")) {
        show_labels = j["show_labels"].get<bool>();
    }
    if (j.contains("keyboard")) {
        keyboard = j["keyboard"].get<bool>();
    }
    if (j.contains("savevideo")) {
        savevideo = j["savevideo"].get<bool>();
    }
    if (j.contains("rtprio")) {
        rtprio = j["rtprio"].get<bool>();
    }
    if (j.contains("beadsearch_R")) {
        beadsearch_R = j["beadsearch_R"].get<bool>();
    }
    if (j.contains("beadsearch_G")) {
        beadsearch_G = j["beadsearch_G"].get<bool>();
    }
    if (j.contains("tcp_port")) {
        tcp_port = j["tcp_port"].get<int>();
    }
    

    if (j.contains("img_width")) {
        img_width = j["img_width"].get<int>();
    }
    if (j.contains("img_height")) {
        img_height = j["img_height"].get<int>();
    }
    if (j.contains("img_offset_X")) {
        img_offset_X = j["img_offset_X"].get<int>();
    }
    if (j.contains("img_offset_Y")) {
        img_offset_Y = j["img_offset_Y"].get<int>();
    }
    if (j.contains("img_offset_R2G_X")) {
        img_offset_R2G_X = j["img_offset_R2G_X"].get<int>();
    }
    if (j.contains("img_offset_R2G_Y")) {
        img_offset_R2G_Y = j["img_offset_R2G_Y"].get<int>();
    }
    if (j.contains("cam_exposure")) {
        cam_exposure = j["cam_exposure"].get<int>();
    }
    if (j.contains("cam_analoggain")) {
        cam_analoggain = j["cam_analoggain"].get<int>();
    }
    if (j.contains("cam_digitalgain")) {
        cam_digitalgain = j["cam_digitalgain"].get<int>();
    }
    if (j.contains("cam_FPS")) {
        cam_FPS = j["cam_FPS"].get<int>();
    }
    if (j.contains("backprop_z_R")) {
        backprop_z_R = j["backprop_z_R"].get<int>();
    }
    if (j.contains("backprop_z_G")) {
        backprop_z_G = j["backprop_z_G"].get<int>();
    }
    if (j.contains("improc_thrs_G")) {
        improc_thrs_G = j["improc_thrs_G"].get<int>();
    }
    if (j.contains("improc_thrs_R")) {
        improc_thrs_R = j["improc_thrs_R"].get<int>();
    }
}

string Params::getJSONConfigString() {
  json j;

  j["verbose"] = verbose;
  j["debug"] = debug;
  j["show"] = show;
  j["show_fullscreen"] = show_fullscreen;
  j["show_markers"] = show_markers;
  j["show_labels"] = show_labels;
  j["keyboard"] = keyboard;
  j["savevideo"] = savevideo;
  j["rtprio"] = rtprio;
  j["beadsearch_R"] = beadsearch_R;
  j["beadsearch_G"] = beadsearch_G;

  j["tcp_port"] = tcp_port;
  j["img_width"] = img_width;
  j["img_height"] = img_height;
  j["img_offset_X"] = img_offset_X;
  j["img_offset_Y"] = img_offset_Y;
  j["img_offset_R2G_X"] = img_offset_R2G_X;
  j["img_offset_R2G_Y"] = img_offset_R2G_Y;
  j["cam_exposure"] = cam_exposure;
  j["cam_analoggain"] = cam_analoggain;
  j["cam_digitalgain"] = cam_digitalgain;
  j["cam_FPS"] = cam_FPS;
  j["backprop_z_R"] = backprop_z_R;
  j["backprop_z_G"] = backprop_z_G;
  j["improc_thrs_G"] = improc_thrs_G;
  j["improc_thrs_R"] = improc_thrs_R;

  return j.dump();
}

void Params::parseCmdlineArgs(int argc, char* argv[])
{
  try
  {
    cxxopts::Options options(argv[0], " - Twin-beam setup - image processing");
    options
      .positional_help("[optional args]")
      .show_positional_help();

    options
      .add_options()
      ("s,show", 		"Display the processed image on the display",				cxxopts::value<bool>())
      ("fullscreen",	"Display the processed image on the display in fullscreen. '-s' flag must be used.",	cxxopts::value<bool>())
      ("showmarkers", 	"Display markers at the positions of found/tracked objects. '-s' flag must be used.",cxxopts::value<bool>())
      ("showlabels", 	"Display labels at the positions of tracked objects. '-s' flag must be used.",		cxxopts::value<bool>())
      ("savevideo", 	"Save video - works only if 'show' argument is used as well",cxxopts::value<bool>())
      ("d,debug", 		"Prints debug information",									cxxopts::value<bool>())
      ("k,keyboard", 	"Enable keyboard input",									cxxopts::value<bool>())
      ("v,verbose", 	"Prints some additional information",						cxxopts::value<bool>())
      ("p,rtprio", 		"Set real-time priorities",									cxxopts::value<bool>())
      ("tcpport", 		"TCP port of the server",									cxxopts::value<uint32_t>())
      ("help", 			"Prints help")
	  ;
	  
	options.add_options("Camera")
      ("e,exp",			"Exposure time (us) [8,333333]",							cxxopts::value<uint32_t>())
	  ("analoggain", 	"Analog gain [1,354]", 										cxxopts::value<uint32_t>())
	  ("digitalgain", 	"Digital gain [1,256]", 									cxxopts::value<uint32_t>())
      ("r,resolution", 	"Resolution (example -r 1024,1024)",						cxxopts::value<std::vector<uint32_t>>())
	  ("o,offset", 		"Offset of the image (example -o 123,523)", 				cxxopts::value<std::vector<uint32_t>>())
	  ("f,fps", 		"Frame rate [1,50]", 										cxxopts::value<uint32_t>())
	  ;
	  
	options.add_options("Image Processing")
      ("imthrs_g", 		"Upper threshold for the green channel",					cxxopts::value<uint32_t>())
      ("imthrs_r", 		"Prints debug information",									cxxopts::value<uint32_t>())
      ("g_dist", 		"Green channel backpropagation distance",					cxxopts::value<uint32_t>())
      ("r_dist", 		"Red channel backpropagation distance",						cxxopts::value<uint32_t>())
      ("b,beadsearch", 	"Enable searching the beads in the image {none, R, G, RG}",	cxxopts::value<std::string>()->default_value("none")->implicit_value("RG"))
	  ;
	
    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
      std::cout << options.help({"", "Camera", "Image Processing"}) << std::endl;
      exit(0);
	}

    if (result.count("verbose") > 0)
		verbose	= result["verbose"].as<bool>();
    if (result.count("debug") > 0)
		debug	= result["debug"].as<bool>();
    if (result.count("show") > 0)
		show	= result["show"].as<bool>();
    if (result.count("show_fullscreen") > 0)
		show_fullscreen	= result["show_fullscreen"].as<bool>();
    if (result.count("show_markers") > 0)
		show_markers	= result["show_markers"].as<bool>();
    if (result.count("show_labels") > 0)
		show_labels	= result["show_labels"].as<bool>();
    if (result.count("keyboard") > 0)
		keyboard	= result["keyboard"].as<bool>();
    if (result.count("savevideo") > 0)
		savevideo	= result["savevideo"].as<bool>();
    if (result.count("rtprio") > 0)
		rtprio	= result["rtprio"].as<bool>();
    if (result.count("beadsearch_R") > 0)
		beadsearch_R	= result["beadsearch_R"].as<bool>();
    if (result.count("beadsearch_G") > 0)
		beadsearch_G	= result["beadsearch_G"].as<bool>();

	if (result.count("tcpport") > 0)
		tcp_port	= result["tcpport"].as<uint16_t>();
	if (result.count("exp") > 0)
		cam_exposure	= result["exp"].as<uint32_t>()*1e3;
	if (result.count("digitalgain") > 0)
		cam_digitalgain	= result["digitalgain"].as<uint32_t>();
	if (result.count("analoggain") > 0)
		cam_analoggain = result["analoggain"].as<uint32_t>();

	if (result.count("fps") > 0)
		cam_FPS = result["fps"].as<uint32_t>();

	if (result.count("resolution") > 0) {
		const auto values = result["resolution"].as<std::vector<uint32_t>>();
		img_width 	= values[0];
		img_height 	= values[1];
	}
	if (result.count("offset") > 0) {
		const auto values = result["offset"].as<std::vector<uint32_t>>();
		img_offset_X 	= values[0];
		img_offset_Y 	= values[1];
	}		
	
	if (result.count("imthrs_g") > 0) {
		improc_thrs_G = result["imthrs_g"].as<uint32_t>();
	}
	if (result.count("imthrs_r") > 0) {
		improc_thrs_R = result["imthrs_r"].as<uint32_t>();
	}
	if (result.count("r_dist") > 0) {
		backprop_z_R= result["r_dist"].as<uint32_t>();
	}
	if (result.count("g_dist") > 0) {
		backprop_z_G= result["g_dist"].as<uint32_t>();
	}
	if (result.count("beadsearch") > 0) {
		if (result["beadsearch"].as<std::string>().compare("R") == 0) {
			beadsearch_R	= true;
			beadsearch_G	= false;
			if (debug) std::cout << "Tracker: R channel" << std::endl;
		} else if (result["beadsearch"].as<std::string>().compare("G") == 0) {
			beadsearch_R	= false;
			beadsearch_G	= true;
			if (debug) std::cout << "Tracker: G channel" << std::endl;
		} else if (result["beadsearch"].as<std::string>().compare("RG") == 0) {
			beadsearch_R	= true;
			beadsearch_G	= true;
			if (debug) std::cout << "Tracker: RG channel" << std::endl;
		} else {
			beadsearch_R	= false;
			beadsearch_G	= false;
			if (debug) std::cout << "Tracker: none channel" << std::endl;
		}
	}

  } catch (const cxxopts::OptionException& e)
  {
    std::cout << "error parsing options: " << e.what() << std::endl;
    exit(1);
  }
}