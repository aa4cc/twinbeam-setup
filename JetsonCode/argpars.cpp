/**
 * @author  Martin Gurtner
 */
#include "argpars.h"

bool Options::verbose 		= false;
bool Options::debug 		= false;
bool Options::show 			= false;
bool Options::saveimgs 		= false;
bool Options::savevideo 	= false;
bool Options::mousekill 	= false;
bool Options::rtprio		= false;
bool Options::beadsearch	= false;

cxxopts::ParseResult Options::parse(AppData& appData, int argc, char* argv[])
{
  try
  {
    cxxopts::Options options(argv[0], " - Twin-beam setup - image processing");
    options
      .positional_help("[optional args]")
      .show_positional_help();

    options
      .add_options()
      ("s,show", 		"Display the processed image on the display",				cxxopts::value<bool>(Options::show))
      ("saveimgs", 		"Save images", 												cxxopts::value<bool>(Options::saveimgs))
      ("savevideo", 	"Save video - works only if 'show' argument is used as well",cxxopts::value<bool>(Options::savevideo))
      ("d,debug", 		"Prints debug information",									cxxopts::value<bool>(Options::debug))
      ("k,mousekill", 	"Moving the mouse or toching the screen kills the app",		cxxopts::value<bool>(Options::mousekill))
      ("v,verbose", 	"Prints some additional information",						cxxopts::value<bool>(Options::verbose))
      ("p,rtprio", 		"Set real-time priorities",									cxxopts::value<bool>(Options::rtprio))
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
      ("imthrs_g", 		"Upper threshold for the green channel",							cxxopts::value<uint32_t>())
      ("imthrs_r", 		"Prints debug information",							cxxopts::value<uint32_t>())
      ("g_dist", 		"Green channel backpropagation distance",			cxxopts::value<uint32_t>())
      ("r_dist", 		"Red channel backpropagation distance",				cxxopts::value<uint32_t>())
      ("b,beadsearch", 	"Enable searching the beads in the image",			cxxopts::value<bool>(Options::beadsearch))
	  ;
	
    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
      std::cout << options.help({"", "Camera", "Image Processing"}) << std::endl;
      exit(0);
	}
	
	if (result.count("exp") > 0)
		appData.values[STG_EXPOSURE] 	= result["exp"].as<uint32_t>()*1e3;
	if (result.count("digitalgain") > 0)
		appData.values[STG_DIGGAIN] 	= result["digitalgain"].as<uint32_t>();
	if (result.count("analoggain") > 0)
		appData.values[STG_ANALOGGAIN]= result["analoggain"].as<uint32_t>();

	if (result.count("fps") > 0)
		appData.values[STG_FPS]= result["fps"].as<uint32_t>();

	if (result.count("resolution") > 0) {
		const auto values = result["resolution"].as<std::vector<uint32_t>>();
		appData.values[STG_WIDTH] 	= values[0];
		appData.values[STG_HEIGHT] 	= values[1];
	}
	if (result.count("offset") > 0) {
		const auto values = result["offset"].as<std::vector<uint32_t>>();
		appData.values[STG_OFFSET_X] 	= values[0];
		appData.values[STG_OFFSET_Y] 	= values[1];
	}		
	
	if (result.count("imthrs_g") > 0) {
		appData.values[STG_IMGTHRS_G]= result["imthrs_g"].as<uint32_t>();
	}
	if (result.count("imthrs_r") > 0) {
		appData.values[STG_IMGTHRS_R]= result["imthrs_r"].as<uint32_t>();
	}
	if (result.count("r_dist") > 0) {
		appData.values[STG_Z_RED]= result["r_dist"].as<uint32_t>();
	}
	if (result.count("g_dist") > 0) {
		appData.values[STG_Z_GREEN]= result["g_dist"].as<uint32_t>();
	}

    return result;

  } catch (const cxxopts::OptionException& e)
  {
    std::cout << "error parsing options: " << e.what() << std::endl;
    exit(1);
  }
}