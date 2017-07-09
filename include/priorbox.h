#ifndef PRIORBOX_H_
#define PRIORBOX_H_

#include "alone_net.h"
#include "network.h"

void make_priorbox(float *output,
                     int layer_h, int layer_w, int net_h, int net_w,
                     float min_size, float max_size, int use_3_aspect);

#endif
