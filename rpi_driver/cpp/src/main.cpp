#include "led.hpp"

namespace rpi = yrgo::rpi;

// -----------------------------------------------------------------------------
int main()
{
    rpi::Led led1{17};
    
    while (1)
    {
        led1.blink(100);
    }
    return 0;
}