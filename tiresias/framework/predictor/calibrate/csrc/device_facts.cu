// Prints the live device facts as one JSON object (no kernel is launched). Used by run_calibration.py to build the plan and to cross-check HARDWARE_GROUND_TRUTH.md.
#include "cal_common.h"
int main() { DeviceInfo d = cal_init_device(); cal_print_device(d); return 0; }
