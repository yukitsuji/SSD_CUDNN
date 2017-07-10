#include  "priorbox.h"
#include <math.h>

// HWP4
void make_priorbox(float *output,
                     int layer_h, int layer_w, int net_h, int net_w,
                     float min_size, float max_size, int use_3_aspect){
  int step_w, step_h;
  float box_width, box_height;
  int aspect_ratio;
  int h, w;
  step_w = net_w / layer_w;
  step_h = net_h / layer_h;

  int idx = 0;
  for (h = 0; h < layer_h; ++h) {
    for (w = 0; w < layer_w; ++w) {
      float center_x = (w + 0.5) * step_w;
      float center_y = (h + 0.5) * step_h;

      // first prior: aspect_ratio = 1, size = min_size
      box_width = box_height = min_size;
      // x
      output[idx++] = (center_x) / net_w;
      // y
      output[idx++] = (center_y) / net_h;
      // width
      output[idx++] = (box_width) / net_w;
      // height
      output[idx++] = (box_height) / net_h;

      // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
      box_width = box_height = sqrt(min_size * max_size);
      // x
      output[idx++] = (center_x) / net_w;
      // y
      output[idx++] = (center_y) / net_h;
      // width
      output[idx++] = (box_width) / net_w;
      // height
      output[idx++] = (box_height) / net_h;

      aspect_ratio = 2;
      box_width = min_size * sqrt(aspect_ratio);
      box_height = min_size / sqrt(aspect_ratio);
      // x
      output[idx++] = (center_x) / net_w;
      // y
      output[idx++] = (center_y) / net_h;
      // width
      output[idx++] = (box_width) / net_w;
      // height
      output[idx++] = (box_height) / net_h;

      if (use_3_aspect == 0) continue;

      aspect_ratio = 3;
      box_width = min_size * sqrt(aspect_ratio);
      box_height = min_size / sqrt(aspect_ratio);
      // x
      output[idx++] = (center_x) / net_w;
      // y
      output[idx++] = (center_y) / net_h;
      // width
      output[idx++] = (box_width) / net_w;
      // height
      output[idx++] = (box_height) / net_h;
    }
  }
}
