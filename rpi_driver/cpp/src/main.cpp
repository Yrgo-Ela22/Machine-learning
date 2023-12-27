#include "button.hpp"
#include "led.hpp"

namespace rpi = yrgo::rpi;

// -----------------------------------------------------------------------------
int main()
{
    rpi::Led led1{17};
    rpi::Button button1{27};
    
    while (1)
    {
        if (button1.isEventDetected())
        {
            led1.toggle();
        }
    }
    return 0;
}