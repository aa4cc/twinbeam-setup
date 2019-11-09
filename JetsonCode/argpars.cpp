#include "argpars.h"

bool Options::verbose 	= false;
bool Options::debug 		= false;
bool Options::show 		= false;
bool Options::saveimgs 	= false;
bool Options::mousekill 	= false;

cxxopts::ParseResult Options::parse(int argc, char* argv[])
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
      ("d,debug", 		"Prints debug information",									cxxopts::value<bool>(Options::debug))
      ("k,mousekill", 	"Moving the mouse or toching the screen kills the app",		cxxopts::value<bool>(Options::mousekill))
      ("v,verbose", 	"Prints some additional information",						cxxopts::value<bool>(Options::verbose))
      ("help", 			"Prints help")
	  ;
	  
	options.add_options("Camera")
      ("e,exp",			"Exposure time (us) [8,333333]",								cxxopts::value<uint32_t>())
	  ("analoggain", 	"Analog gain [1,354]", 										cxxopts::value<uint32_t>())
	  ("digitalgain", 	"Digital gain [1,256]", 										cxxopts::value<uint32_t>())
      ("r,resolution", 	"Resolution (example -r 1024,1024)",						cxxopts::value<std::vector<uint32_t>>())
	  ("o,offset", 		"Offset of the image (example -o 123,523)", 				cxxopts::value<std::vector<uint32_t>>())
	  ("f,fps", 		"Frame rate [1,60]", 												cxxopts::value<uint32_t>())
	  ;
	
    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
      std::cout << options.help({"", "Camera"}) << std::endl;
      exit(0);
	}
	
	if (result.count("exp") > 0)
		Settings::values[STG_EXPOSURE] 	= result["exp"].as<uint32_t>()*1e3;
	if (result.count("digitalgain") > 0)
		Settings::values[STG_DIGGAIN] 	= result["digitalgain"].as<uint32_t>();
	if (result.count("analoggain") > 0)
		Settings::values[STG_ANALOGGAIN]= result["analoggain"].as<uint32_t>();

	if (result.count("fps") > 0)
		Settings::values[STG_FPS]= result["fps"].as<uint32_t>();

	if (result.count("resolution") > 0) {
		const auto values = result["resolution"].as<std::vector<uint32_t>>();
		Settings::values[STG_WIDTH] 	= values[0];
		Settings::values[STG_HEIGHT] 	= values[1];
	}
	if (result.count("offset") > 0) {
		const auto values = result["offset"].as<std::vector<uint32_t>>();
		Settings::values[STG_OFFSET_X] 	= values[0];
		Settings::values[STG_OFFSET_Y] 	= values[1];
	}		

    if (Options::debug) {
	    if (Options::show)
	    {
	      std::cout << "Saw option ‘s’" << std::endl;
	    }

	    if (Options::debug)
	    {
	      std::cout << "Saw option ‘d’" << std::endl;
	    }

	    if (Options::verbose)
	    {
	      std::cout << "Saw option ‘v’" << std::endl;
	    }
	}


    return result;

  } catch (const cxxopts::OptionException& e)
  {
    std::cout << "error parsing options: " << e.what() << std::endl;
    exit(1);
  }
}