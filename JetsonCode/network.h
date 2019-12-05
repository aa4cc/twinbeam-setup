#ifndef NETWORK_H
#define NETWORK_H

#include "AppData.h"

void network_thread(AppData& appData);
void datasend_thread(AppData& appData);

#endif