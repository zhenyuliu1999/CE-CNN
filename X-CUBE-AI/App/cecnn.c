/**
  ******************************************************************************
  * @file    cecnn.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Sun Jun 16 21:36:44 2024
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "cecnn.h"
#include "cecnn_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_cecnn
 
#undef AI_CECNN_MODEL_SIGNATURE
#define AI_CECNN_MODEL_SIGNATURE     "f12a0b86a0ed901df047c8424a65705d"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "Sun Jun 16 21:36:44 2024"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_CECNN_N_BATCHES
#define AI_CECNN_N_BATCHES         (1)

static ai_ptr g_cecnn_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_cecnn_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  input_1_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 2520, AI_STATIC)
/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  conv1d_conv2d_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1440, AI_STATIC)
/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  conv1d_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1440, AI_STATIC)
/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  batch_normalization_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1440, AI_STATIC)
/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  max_pooling1d_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1440, AI_STATIC)
/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  global_average_pooling1d_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  dense_2_dense_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  dense_2_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  dense_3_dense_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  dense_3_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  dense_dense_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)
/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  dense_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)
/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  dense_1_dense_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  dense_1_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  multiply_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1440, AI_STATIC)
/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  add_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1440, AI_STATIC)
/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  dense_4_dense_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)
/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  dense_4_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)
/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  dense_5_dense_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 1, AI_STATIC)
/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  conv1d_conv2d_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 448, AI_STATIC)
/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  conv1d_conv2d_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  batch_normalization_scale_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  batch_normalization_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  dense_2_dense_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 256, AI_STATIC)
/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  dense_2_dense_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  dense_3_dense_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 256, AI_STATIC)
/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  dense_3_dense_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  dense_dense_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)
/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  dense_dense_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)
/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  dense_1_dense_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)
/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  dense_1_dense_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)
/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  dense_4_dense_weights_array, AI_ARRAY_FORMAT_LUT8_FLOAT,
  NULL, NULL, 92160, AI_STATIC)
/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  dense_4_dense_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)
/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  dense_5_dense_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)
/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  dense_5_dense_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)
/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  input_1_output, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 360), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &input_1_output_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  conv1d_conv2d_output, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 90), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &conv1d_conv2d_output_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  conv1d_output, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 90), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &conv1d_output_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  batch_normalization_output, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 90), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &batch_normalization_output_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  max_pooling1d_output, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 90), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &max_pooling1d_output_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  global_average_pooling1d_output, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &global_average_pooling1d_output_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  dense_2_dense_output, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &dense_2_dense_output_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  dense_2_output, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &dense_2_output_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  dense_3_dense_output, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &dense_3_dense_output_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  dense_3_output, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &dense_3_output_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  dense_dense_output, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &dense_dense_output_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  dense_output, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &dense_output_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  dense_1_dense_output, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &dense_1_dense_output_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  dense_1_output, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &dense_1_output_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  multiply_output, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 90), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &multiply_output_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  add_output, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 90), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &add_output_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  add_output0, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 1440, 1, 1), AI_STRIDE_INIT(4, 4, 4, 5760, 5760),
  1, &add_output_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  dense_4_dense_output, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &dense_4_dense_output_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  dense_4_output, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &dense_4_output_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  dense_5_dense_output, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &dense_5_dense_output_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  conv1d_conv2d_weights, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 7, 1, 4, 16), AI_STRIDE_INIT(4, 4, 28, 448, 448),
  1, &conv1d_conv2d_weights_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  conv1d_conv2d_bias, AI_STATIC,
  21, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &conv1d_conv2d_bias_array, NULL)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  batch_normalization_scale, AI_STATIC,
  22, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &batch_normalization_scale_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  batch_normalization_bias, AI_STATIC,
  23, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &batch_normalization_bias_array, NULL)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  dense_2_dense_weights, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 16, 16, 1, 1), AI_STRIDE_INIT(4, 4, 64, 1024, 1024),
  1, &dense_2_dense_weights_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  dense_2_dense_bias, AI_STATIC,
  25, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &dense_2_dense_bias_array, NULL)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  dense_3_dense_weights, AI_STATIC,
  26, 0x0,
  AI_SHAPE_INIT(4, 16, 16, 1, 1), AI_STRIDE_INIT(4, 4, 64, 1024, 1024),
  1, &dense_3_dense_weights_array, NULL)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  dense_3_dense_bias, AI_STATIC,
  27, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &dense_3_dense_bias_array, NULL)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  dense_dense_weights, AI_STATIC,
  28, 0x0,
  AI_SHAPE_INIT(4, 16, 8, 1, 1), AI_STRIDE_INIT(4, 4, 64, 512, 512),
  1, &dense_dense_weights_array, NULL)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  dense_dense_bias, AI_STATIC,
  29, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &dense_dense_bias_array, NULL)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  dense_1_dense_weights, AI_STATIC,
  30, 0x0,
  AI_SHAPE_INIT(4, 8, 16, 1, 1), AI_STRIDE_INIT(4, 4, 32, 512, 512),
  1, &dense_1_dense_weights_array, NULL)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  dense_1_dense_bias, AI_STATIC,
  31, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &dense_1_dense_bias_array, NULL)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  dense_4_dense_weights, AI_STATIC,
  32, 0x0,
  AI_SHAPE_INIT(4, 1440, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1440, 92160, 92160),
  1, &dense_4_dense_weights_array, NULL)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  dense_4_dense_bias, AI_STATIC,
  33, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &dense_4_dense_bias_array, NULL)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  dense_5_dense_weights, AI_STATIC,
  34, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 1, 1), AI_STRIDE_INIT(4, 4, 256, 256, 256),
  1, &dense_5_dense_weights_array, NULL)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  dense_5_dense_bias, AI_STATIC,
  35, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &dense_5_dense_bias_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_5_dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_5_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &dense_5_dense_weights, &dense_5_dense_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_5_dense_layer, 15,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &dense_5_dense_chain,
  NULL, &dense_5_dense_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_4_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_4_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_4_layer, 14,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &dense_4_chain,
  NULL, &dense_5_dense_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_4_dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &add_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_4_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &dense_4_dense_weights, &dense_4_dense_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_4_dense_layer, 14,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &dense_4_dense_chain,
  NULL, &dense_4_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  add_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &multiply_output, &dense_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &add_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  add_layer, 12,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &add_chain,
  NULL, &dense_4_dense_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  multiply_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &max_pooling1d_output, &dense_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &multiply_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  multiply_layer, 10,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &multiply_chain,
  NULL, &add_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_1_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_1_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_1_layer, 7,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &dense_1_chain,
  NULL, &multiply_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_1_dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_1_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &dense_1_dense_weights, &dense_1_dense_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_1_dense_layer, 7,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &dense_1_dense_chain,
  NULL, &dense_1_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_layer, 6,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &dense_chain,
  NULL, &dense_1_dense_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &global_average_pooling1d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &dense_dense_weights, &dense_dense_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_dense_layer, 6,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &dense_dense_chain,
  NULL, &dense_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_3_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_3_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_3_layer, 11,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &dense_3_chain,
  NULL, &dense_dense_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_3_dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_3_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &dense_3_dense_weights, &dense_3_dense_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_3_dense_layer, 11,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &dense_3_dense_chain,
  NULL, &dense_3_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_2_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_2_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_2_layer, 9,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &dense_2_chain,
  NULL, &dense_3_dense_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_2_dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &global_average_pooling1d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_2_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &dense_2_dense_weights, &dense_2_dense_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_2_dense_layer, 9,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &dense_2_dense_chain,
  NULL, &dense_2_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  global_average_pooling1d_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &max_pooling1d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &global_average_pooling1d_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  global_average_pooling1d_layer, 4,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap,
  &global_average_pooling1d_chain,
  NULL, &dense_2_dense_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 90), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 90), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  max_pooling1d_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &batch_normalization_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &max_pooling1d_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  max_pooling1d_layer, 3,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp,
  &max_pooling1d_chain,
  NULL, &global_average_pooling1d_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 1), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 1), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  batch_normalization_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv1d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &batch_normalization_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &batch_normalization_scale, &batch_normalization_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  batch_normalization_layer, 2,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &batch_normalization_chain,
  NULL, &max_pooling1d_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv1d_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv1d_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv1d_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv1d_layer, 1,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &conv1d_chain,
  NULL, &batch_normalization_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv1d_conv2d_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv1d_conv2d_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv1d_conv2d_weights, &conv1d_conv2d_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv1d_conv2d_layer, 1,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv1d_conv2d_chain,
  NULL, &conv1d_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 4), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 98980, 1, 1),
    98980, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 10180, 1, 1),
    10180, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_CECNN_IN_NUM, &input_1_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_CECNN_OUT_NUM, &dense_5_dense_output),
  &conv1d_conv2d_layer, 0, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 98980, 1, 1),
      98980, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 10180, 1, 1),
      10180, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_CECNN_IN_NUM, &input_1_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_CECNN_OUT_NUM, &dense_5_dense_output),
  &conv1d_conv2d_layer, 0, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/


/******************************************************************************/
AI_DECLARE_STATIC
ai_bool cecnn_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_cecnn_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    input_1_output_array.data = AI_PTR(g_cecnn_activations_map[0] + 100);
    input_1_output_array.data_start = AI_PTR(g_cecnn_activations_map[0] + 100);
    
    conv1d_conv2d_output_array.data = AI_PTR(g_cecnn_activations_map[0] + 0);
    conv1d_conv2d_output_array.data_start = AI_PTR(g_cecnn_activations_map[0] + 0);
    
    conv1d_output_array.data = AI_PTR(g_cecnn_activations_map[0] + 0);
    conv1d_output_array.data_start = AI_PTR(g_cecnn_activations_map[0] + 0);
    
    batch_normalization_output_array.data = AI_PTR(g_cecnn_activations_map[0] + 0);
    batch_normalization_output_array.data_start = AI_PTR(g_cecnn_activations_map[0] + 0);
    
    max_pooling1d_output_array.data = AI_PTR(g_cecnn_activations_map[0] + 0);
    max_pooling1d_output_array.data_start = AI_PTR(g_cecnn_activations_map[0] + 0);
    
    global_average_pooling1d_output_array.data = AI_PTR(g_cecnn_activations_map[0] + 5760);
    global_average_pooling1d_output_array.data_start = AI_PTR(g_cecnn_activations_map[0] + 5760);
    
    dense_2_dense_output_array.data = AI_PTR(g_cecnn_activations_map[0] + 5824);
    dense_2_dense_output_array.data_start = AI_PTR(g_cecnn_activations_map[0] + 5824);
    
    dense_2_output_array.data = AI_PTR(g_cecnn_activations_map[0] + 5888);
    dense_2_output_array.data_start = AI_PTR(g_cecnn_activations_map[0] + 5888);
    
    dense_3_dense_output_array.data = AI_PTR(g_cecnn_activations_map[0] + 5824);
    dense_3_dense_output_array.data_start = AI_PTR(g_cecnn_activations_map[0] + 5824);
    
    dense_3_output_array.data = AI_PTR(g_cecnn_activations_map[0] + 5888);
    dense_3_output_array.data_start = AI_PTR(g_cecnn_activations_map[0] + 5888);
    
    dense_dense_output_array.data = AI_PTR(g_cecnn_activations_map[0] + 5824);
    dense_dense_output_array.data_start = AI_PTR(g_cecnn_activations_map[0] + 5824);
    
    dense_output_array.data = AI_PTR(g_cecnn_activations_map[0] + 5760);
    dense_output_array.data_start = AI_PTR(g_cecnn_activations_map[0] + 5760);
    
    dense_1_dense_output_array.data = AI_PTR(g_cecnn_activations_map[0] + 5792);
    dense_1_dense_output_array.data_start = AI_PTR(g_cecnn_activations_map[0] + 5792);
    
    dense_1_output_array.data = AI_PTR(g_cecnn_activations_map[0] + 5952);
    dense_1_output_array.data_start = AI_PTR(g_cecnn_activations_map[0] + 5952);
    
    multiply_output_array.data = AI_PTR(g_cecnn_activations_map[0] + 0);
    multiply_output_array.data_start = AI_PTR(g_cecnn_activations_map[0] + 0);
    
    add_output_array.data = AI_PTR(g_cecnn_activations_map[0] + 0);
    add_output_array.data_start = AI_PTR(g_cecnn_activations_map[0] + 0);
    
    dense_4_dense_output_array.data = AI_PTR(g_cecnn_activations_map[0] + 5760);
    dense_4_dense_output_array.data_start = AI_PTR(g_cecnn_activations_map[0] + 5760);
    
    dense_4_output_array.data = AI_PTR(g_cecnn_activations_map[0] + 0);
    dense_4_output_array.data_start = AI_PTR(g_cecnn_activations_map[0] + 0);
    
    dense_5_dense_output_array.data = AI_PTR(g_cecnn_activations_map[0] + 256);
    dense_5_dense_output_array.data_start = AI_PTR(g_cecnn_activations_map[0] + 256);
    
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool cecnn_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_cecnn_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    conv1d_conv2d_weights_array.format |= AI_FMT_FLAG_CONST;
    conv1d_conv2d_weights_array.data = AI_PTR(g_cecnn_weights_map[0] + 0);
    conv1d_conv2d_weights_array.data_start = AI_PTR(g_cecnn_weights_map[0] + 0);
    
    conv1d_conv2d_bias_array.format |= AI_FMT_FLAG_CONST;
    conv1d_conv2d_bias_array.data = AI_PTR(g_cecnn_weights_map[0] + 1792);
    conv1d_conv2d_bias_array.data_start = AI_PTR(g_cecnn_weights_map[0] + 1792);
    
    batch_normalization_scale_array.format |= AI_FMT_FLAG_CONST;
    batch_normalization_scale_array.data = AI_PTR(g_cecnn_weights_map[0] + 1856);
    batch_normalization_scale_array.data_start = AI_PTR(g_cecnn_weights_map[0] + 1856);
    
    batch_normalization_bias_array.format |= AI_FMT_FLAG_CONST;
    batch_normalization_bias_array.data = AI_PTR(g_cecnn_weights_map[0] + 1920);
    batch_normalization_bias_array.data_start = AI_PTR(g_cecnn_weights_map[0] + 1920);
    
    dense_2_dense_weights_array.format |= AI_FMT_FLAG_CONST;
    dense_2_dense_weights_array.data = AI_PTR(g_cecnn_weights_map[0] + 1984);
    dense_2_dense_weights_array.data_start = AI_PTR(g_cecnn_weights_map[0] + 1984);
    
    dense_2_dense_bias_array.format |= AI_FMT_FLAG_CONST;
    dense_2_dense_bias_array.data = AI_PTR(g_cecnn_weights_map[0] + 3008);
    dense_2_dense_bias_array.data_start = AI_PTR(g_cecnn_weights_map[0] + 3008);
    
    dense_3_dense_weights_array.format |= AI_FMT_FLAG_CONST;
    dense_3_dense_weights_array.data = AI_PTR(g_cecnn_weights_map[0] + 3072);
    dense_3_dense_weights_array.data_start = AI_PTR(g_cecnn_weights_map[0] + 3072);
    
    dense_3_dense_bias_array.format |= AI_FMT_FLAG_CONST;
    dense_3_dense_bias_array.data = AI_PTR(g_cecnn_weights_map[0] + 4096);
    dense_3_dense_bias_array.data_start = AI_PTR(g_cecnn_weights_map[0] + 4096);
    
    dense_dense_weights_array.format |= AI_FMT_FLAG_CONST;
    dense_dense_weights_array.data = AI_PTR(g_cecnn_weights_map[0] + 4160);
    dense_dense_weights_array.data_start = AI_PTR(g_cecnn_weights_map[0] + 4160);
    
    dense_dense_bias_array.format |= AI_FMT_FLAG_CONST;
    dense_dense_bias_array.data = AI_PTR(g_cecnn_weights_map[0] + 4672);
    dense_dense_bias_array.data_start = AI_PTR(g_cecnn_weights_map[0] + 4672);
    
    dense_1_dense_weights_array.format |= AI_FMT_FLAG_CONST;
    dense_1_dense_weights_array.data = AI_PTR(g_cecnn_weights_map[0] + 4704);
    dense_1_dense_weights_array.data_start = AI_PTR(g_cecnn_weights_map[0] + 4704);
    
    dense_1_dense_bias_array.format |= AI_FMT_FLAG_CONST;
    dense_1_dense_bias_array.data = AI_PTR(g_cecnn_weights_map[0] + 5216);
    dense_1_dense_bias_array.data_start = AI_PTR(g_cecnn_weights_map[0] + 5216);
    
    dense_4_dense_weights_array.format |= AI_FMT_FLAG_CONST;
    dense_4_dense_weights_array.data = AI_PTR(g_cecnn_weights_map[0] + 6304);
    dense_4_dense_weights_array.data_start = AI_PTR(g_cecnn_weights_map[0] + 5280);
    
    dense_4_dense_bias_array.format |= AI_FMT_FLAG_CONST;
    dense_4_dense_bias_array.data = AI_PTR(g_cecnn_weights_map[0] + 98464);
    dense_4_dense_bias_array.data_start = AI_PTR(g_cecnn_weights_map[0] + 98464);
    
    dense_5_dense_weights_array.format |= AI_FMT_FLAG_CONST;
    dense_5_dense_weights_array.data = AI_PTR(g_cecnn_weights_map[0] + 98720);
    dense_5_dense_weights_array.data_start = AI_PTR(g_cecnn_weights_map[0] + 98720);
    
    dense_5_dense_bias_array.format |= AI_FMT_FLAG_CONST;
    dense_5_dense_bias_array.data = AI_PTR(g_cecnn_weights_map[0] + 98976);
    dense_5_dense_bias_array.data_start = AI_PTR(g_cecnn_weights_map[0] + 98976);
    
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/


AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_cecnn_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_CECNN_MODEL_NAME,
      .model_signature   = AI_CECNN_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 144009,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x0,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_bool ai_cecnn_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_CECNN_MODEL_NAME,
      .model_signature   = AI_CECNN_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 144009,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x0,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}

AI_API_ENTRY
ai_error ai_cecnn_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}

AI_API_ENTRY
ai_error ai_cecnn_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    &AI_NET_OBJ_INSTANCE,
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}

AI_API_ENTRY
ai_error ai_cecnn_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
    ai_error err;
    ai_network_params params;

    err = ai_cecnn_create(network, AI_CECNN_DATA_CONFIG);
    if (err.type != AI_ERROR_NONE)
        return err;
    if (ai_cecnn_data_params_get(&params) != true) {
        err = ai_cecnn_get_error(*network);
        return err;
    }
#if defined(AI_CECNN_DATA_ACTIVATIONS_COUNT)
    if (activations) {
        /* set the addresses of the activations buffers */
        for (int idx=0;idx<params.map_activations.size;idx++)
            AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
    }
#endif
#if defined(AI_CECNN_DATA_WEIGHTS_COUNT)
    if (weights) {
        /* set the addresses of the weight buffers */
        for (int idx=0;idx<params.map_weights.size;idx++)
            AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
    }
#endif
    if (ai_cecnn_init(*network, &params) != true) {
        err = ai_cecnn_get_error(*network);
    }
    return err;
}

AI_API_ENTRY
ai_buffer* ai_cecnn_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    ((ai_network *)network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}

AI_API_ENTRY
ai_buffer* ai_cecnn_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    ((ai_network *)network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}

AI_API_ENTRY
ai_handle ai_cecnn_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}

AI_API_ENTRY
ai_bool ai_cecnn_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = ai_platform_network_init(network, params);
  if (!net_ctx) return false;

  ai_bool ok = true;
  ok &= cecnn_configure_weights(net_ctx, params);
  ok &= cecnn_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_cecnn_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}

AI_API_ENTRY
ai_i32 ai_cecnn_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_CECNN_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

