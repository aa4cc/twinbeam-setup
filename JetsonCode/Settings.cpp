#include "Settings.h"

using namespace std;

int Settings::values[] = {1024, 1024, 1195, 500, 5000000, 3100, 2750};

void Settings::set_setting(int index, const int new_setting){
	Settings::values[index] = new_setting;
}
