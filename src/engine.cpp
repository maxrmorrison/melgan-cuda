#include "cuda.hpp"
#include "model.hpp"


/******************************************************************************
Execution engine
******************************************************************************/


namespace engine {
    /* start the engine */
    void start()
    {
        cuda::copy_to_device(CONV_0, false);
        cuda::copy_to_device(CONV_1, true);
        cuda::copy_to_device(CONV_2, false);
        cuda::copy_to_device(CONV_3, false);
        cuda::copy_to_device(CONV_4, false);
        cuda::copy_to_device(CONV_5, false);
        cuda::copy_to_device(CONV_6, false);
        cuda::copy_to_device(CONV_7, false);
        cuda::copy_to_device(CONV_8, false);
        cuda::copy_to_device(CONV_9, false);
        cuda::copy_to_device(CONV_10, false);
        cuda::copy_to_device(CONV_11, true);
        cuda::copy_to_device(CONV_12, false);
        cuda::copy_to_device(CONV_13, false);
        cuda::copy_to_device(CONV_14, false);
        cuda::copy_to_device(CONV_15, false);
        cuda::copy_to_device(CONV_16, false);
        cuda::copy_to_device(CONV_17, false);
        cuda::copy_to_device(CONV_18, false);
        cuda::copy_to_device(CONV_19, false);
        cuda::copy_to_device(CONV_20, false);
        cuda::copy_to_device(CONV_21, true);
        cuda::copy_to_device(CONV_22, false);
        cuda::copy_to_device(CONV_23, false);
        cuda::copy_to_device(CONV_24, false);
        cuda::copy_to_device(CONV_25, false);
        cuda::copy_to_device(CONV_26, false);
        cuda::copy_to_device(CONV_27, false);
        cuda::copy_to_device(CONV_28, false);
        cuda::copy_to_device(CONV_29, false);
        cuda::copy_to_device(CONV_30, false);
        cuda::copy_to_device(CONV_31, true);
        cuda::copy_to_device(CONV_32, false);
        cuda::copy_to_device(CONV_33, false);
        cuda::copy_to_device(CONV_34, false);
        cuda::copy_to_device(CONV_35, false);
        cuda::copy_to_device(CONV_36, false);
        cuda::copy_to_device(CONV_37, false);
        cuda::copy_to_device(CONV_38, false);
        cuda::copy_to_device(CONV_39, false);
        cuda::copy_to_device(CONV_40, false);
        cuda::copy_to_device(CONV_41, false);
    }


    /* stop the engine */
    void stop()
    {
        cuda::free(CONV_0);
        cuda::free(CONV_1);
        cuda::free(CONV_2);
        cuda::free(CONV_3);
        cuda::free(CONV_4);
        cuda::free(CONV_5);
        cuda::free(CONV_6);
        cuda::free(CONV_7);
        cuda::free(CONV_8);
        cuda::free(CONV_9);
        cuda::free(CONV_10);
        cuda::free(CONV_11);
        cuda::free(CONV_12);
        cuda::free(CONV_13);
        cuda::free(CONV_14);
        cuda::free(CONV_15);
        cuda::free(CONV_16);
        cuda::free(CONV_17);
        cuda::free(CONV_18);
        cuda::free(CONV_19);
        cuda::free(CONV_20);
        cuda::free(CONV_21);
        cuda::free(CONV_22);
        cuda::free(CONV_23);
        cuda::free(CONV_24);
        cuda::free(CONV_25);
        cuda::free(CONV_26);
        cuda::free(CONV_27);
        cuda::free(CONV_28);
        cuda::free(CONV_29);
        cuda::free(CONV_30);
        cuda::free(CONV_31);
        cuda::free(CONV_32);
        cuda::free(CONV_33);
        cuda::free(CONV_34);
        cuda::free(CONV_35);
        cuda::free(CONV_36);
        cuda::free(CONV_37);
        cuda::free(CONV_38);
        cuda::free(CONV_39);
        cuda::free(CONV_40);
        cuda::free(CONV_41);
    }
}
