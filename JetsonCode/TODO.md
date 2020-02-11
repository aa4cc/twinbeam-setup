# Code Chnages
* Avoid using OpenCV Gaussian blur and blur the image in the frequency domain instead
* use opengl for the display
* check the length of the received message when settings are being changed
* add the option to display different images (Backprop, raw{R,G}, filtered)
* get rid of the 'mutexed' access to the beadpos array
* get rid of the blockin behaviour of getchar() in keyboard thread
* add the object to be initialized to std:vector in AppData class and then wait for std::all_of
* Check whether you actually found some beads when the closest positions are sent
* mask individual CPUs to the threads
* Printing to stdout might slower down the app as well thus it should be better to make a separate printinh thread with a circular buffer that would print all the messages
* handle exceptions so that when one thread cannot kill the whole app
* use functor for camera and other threads (?)
* Add the option to start/stop video recording via network
* Check whether BeadTracker is thread-safe or not
* add debug levels
* Unite Options and AppData
* Get rid of Misc.h
* https://github.com/hyperrealm/libconfig or https://github.com/nlohmann/json

sudo sysctl -w kernel.sched_rt_runtime_us=-1