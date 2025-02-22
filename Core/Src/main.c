/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "adc.h"
#include "crc.h"
#include "dma.h"
#include "usart.h"
#include "gpio.h"
#include "fsmc.h"
#include "app_x-cube-ai.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
#include "sys.h"
#include "delay.h"
#include "lcd.h"
#include "dht11.h"
#include "key.h"

ai_float potency1;
unsigned char Data_to_Send[40];

#define BYTE0(temp)	   (*(char*)(&temp))
#define BYTE1(temp)	   (*((char*)(&temp)+1))
#define BYTE2(temp)	   (*((char*)(&temp)+2))
#define BYTE3(temp)	   (*((char*)(&temp)+3))
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
/* Global handle to reference the instantiated C-model */
static ai_handle network = AI_HANDLE_NULL;
 
/* Global c-array to handle the activations buffer */
AI_ALIGNED(32)
static ai_u8 activations[AI_CECNN_DATA_ACTIVATIONS_SIZE];
 
/* Array to store the data of the input tensor */
AI_ALIGNED(32)
static ai_float in_data[AI_CECNN_IN_1_SIZE];
/* or static ai_u8 in_data[AI_HAR_IGN_IN_1_SIZE_BYTES]; */
 
/* c-array to store the data of the output tensor */
AI_ALIGNED(32)
static ai_float out_data[AI_CECNN_OUT_1_SIZE];
/* static ai_u8 out_data[AI_HAR_IGN_OUT_1_SIZE_BYTES]; */
 
/* Array of pointer to manage the model's input/output tensors */
static ai_buffer *ai_input;
static ai_buffer *ai_output;
static ai_buffer_format fmt_input;
static ai_buffer_format fmt_output;


void buf_print(void)
{
	printf("in_data:");
	for (int i=0; i<AI_CECNN_IN_1_SIZE; i++)
	{
		printf("%f ",((ai_float*)in_data)[i]);
	}
	printf("\n");
	printf("out_data:");
	for (int i=0; i<AI_CECNN_OUT_1_SIZE; i++)
	{
		printf("%f ",((ai_float*)out_data)[i]);
	}
	printf("\n");
}
 
void aiPrintBufInfo(const ai_buffer *buffer)
{
	printf("(%lu, %lu, %lu, %lu)", AI_BUFFER_SHAPE_ELEM(buffer, AI_SHAPE_BATCH),
			  	  	  	  	  	  	 AI_BUFFER_SHAPE_ELEM(buffer, AI_SHAPE_HEIGHT),
	                                 AI_BUFFER_SHAPE_ELEM(buffer, AI_SHAPE_WIDTH),
	                                 AI_BUFFER_SHAPE_ELEM(buffer, AI_SHAPE_CHANNEL));
	printf(" buffer_size:%d ", (int)AI_BUFFER_SIZE(buffer));
}
 
void aiPrintDataType(const ai_buffer_format fmt)
{
    if (AI_BUFFER_FMT_GET_TYPE(fmt) == AI_BUFFER_FMT_TYPE_FLOAT)
    	printf("float%d ", (int)AI_BUFFER_FMT_GET_BITS(fmt));
    else if (AI_BUFFER_FMT_GET_TYPE(fmt) == AI_BUFFER_FMT_TYPE_BOOL) {
    	printf("bool%d ", (int)AI_BUFFER_FMT_GET_BITS(fmt));
    } else { /* integer type */
    	printf("%s%d ", AI_BUFFER_FMT_GET_SIGN(fmt)?"i":"u",
            (int)AI_BUFFER_FMT_GET_BITS(fmt));
    }
}
 
void aiPrintDataInfo(const ai_buffer *buffer,const ai_buffer_format fmt)
{
	  if (buffer->data)
		  printf(" @0x%X/%d \n",
	        (int)buffer->data,
	        (int)AI_BUFFER_BYTE_SIZE(AI_BUFFER_SIZE(buffer), fmt)
	    );
	  else
		  printf(" (User Domain)/%d \n",
	        (int)AI_BUFFER_BYTE_SIZE(AI_BUFFER_SIZE(buffer), fmt)
	    );
}
 
void aiPrintNetworkInfo(const ai_network_report report)
{
	printf(" Model name      : %s\r\n", report.model_name);
	printf(" model signature : %s\r\n", report.model_signature);
	printf(" model datetime     : %s\r\n", report.model_datetime);
	printf(" compile datetime   : %s\r\n", report.compile_datetime);
	printf(" runtime version    : %d.%d.%d\r\n",
	      report.runtime_version.major,
	      report.runtime_version.minor,
	      report.runtime_version.micro);
	if (report.tool_revision[0])
		printf(" Tool revision      : %s\r\n", (report.tool_revision[0])?report.tool_revision:"");
	printf(" tools version      : %d.%d.%d\r\n",
	      report.tool_version.major,
	      report.tool_version.minor,
	      report.tool_version.micro);
	printf(" complexity         : %lu MACC\r\n", (unsigned long)report.n_macc);
	printf(" c-nodes            : %d\r\n", (int)report.n_nodes);
 
	printf(" map_activations    : %d\r\n", report.map_activations.size);
	  for (int idx=0; idx<report.map_activations.size;idx++) {
	      const ai_buffer *buffer = &report.map_activations.buffer[idx];
	      printf("  [%d] ", idx);
	      aiPrintBufInfo(buffer);
	      printf("\r\n");
	  }
 
	printf(" map_weights        : %d\r\n", report.map_weights.size);
	  for (int idx=0; idx<report.map_weights.size;idx++) {
	      const ai_buffer *buffer = &report.map_weights.buffer[idx];
	      printf("  [%d] ", idx);
	      aiPrintBufInfo(buffer);
	      printf("\r\n");
	  }
}


int aiInit(void) {
  ai_error err;
 
  /* Create and initialize the c-model */
  const ai_handle acts[] = { activations };
  err = ai_cecnn_create_and_init(&network, acts, NULL);
  if (err.type != AI_ERROR_NONE) {
	  printf("ai_error_type:%d,ai_error_code:%d\r\n",err.type,err.code);
  };
 
  ai_network_report report;
  if (ai_cecnn_get_report(network, &report) != true) {
      printf("ai get report error\n");
      return -1;
  }
 
  aiPrintNetworkInfo(report);
  /* Reteive pointers to the model's input/output tensors */
  ai_input = ai_cecnn_inputs_get(network, NULL);
  ai_output = ai_cecnn_outputs_get(network, NULL);
  //
  fmt_input = AI_BUFFER_FORMAT(ai_input);
  fmt_output = AI_BUFFER_FORMAT(ai_output);
 
  printf(" n_inputs/n_outputs : %u/%u\r\n", report.n_inputs,
            report.n_outputs);
  printf("input :");
  aiPrintBufInfo(ai_input);
  aiPrintDataType(fmt_input);
  aiPrintDataInfo(ai_input, fmt_input);
  
	printf("\r\n");
  printf("output :");
  aiPrintBufInfo(ai_output);
  aiPrintDataType(fmt_output);
  aiPrintDataInfo(ai_output, fmt_output);
	printf("\r\n");
  return 0;
}
ai_float in_buf[2520];

int aiRun(const void *in_data, void *out_data) {
  ai_i32 n_batch;
  ai_error err;
 
  /* 1 - Update IO handlers with the data payload */
  ai_input[0].data = AI_HANDLE_PTR(in_data);
  ai_output[0].data = AI_HANDLE_PTR(out_data);
 
  /* 2 - Perform the inference */
  n_batch = ai_cecnn_run(network, &ai_input[0], &ai_output[0]);
  if (n_batch != 1) {
	  err = ai_cecnn_get_error(network);
	  printf("ai_error_type:%d,ai_error_code:%d\r\n",err.type,err.code);
  };
  return 0;
}

int post_process(void *out_data)
{
	ai_float max_out = 0;
	ai_float potency = 0;
	//printf("\r\npotency:\r\n");
//	for (int i=0; i<AI_CECNN_OUT_1_SIZE; i++)
//	{
////		if((((ai_float*)out_data)[i]) > max_out)
////		{
////			max_out = ((ai_float*)out_data)[i];
////			if(i == 1) potency  = 10;
////			else if(i == 2) potency  = 20;
////			else if(i == 3) potency  = 30;
////			else if(i == 4) potency  = 40;
////			else if(i == 5) potency  = 50;
////			else if(i == 6) potency  = 60;
////			else if(i == 7) potency  = 70;
////			else if(i == 8) potency  = 80;		
////			else if(i == 9) potency  = 90;	
////			else potency  = 100;
////		}
		potency1 = ((ai_float*)out_data)[0];
	//	printf("%f ",((ai_float*)out_data)[i]);
//	}
	//printf("\r\nMaximum_probability:%f \r\n",max_out);
	//printf("potency:%.2f (ppm) \r\n",potency);
	printf("\r\n");
	return 0;
}
int acquire_and_process_data(void *in_data)
{
	
	for (int i=0; i<AI_CECNN_IN_1_SIZE; i++)
	{
		((ai_float*)in_data)[i] =in_buf[i];
		//printf("%.4f ",((ai_float*)in_data)[i]);
		printf("%.4f ",in_buf[i]);
	}
	printf("\r\n");
 
	return 0;
}
void AnoTc_SendUserTest(uint16_t A, uint16_t B, uint16_t C, uint16_t D, uint16_t E, uint16_t F, uint16_t G)
{

	unsigned char _cnt = 0;
	unsigned char i;
	unsigned char sumcheck = 0;
	unsigned char addcheck = 0;

	
	Data_to_Send[_cnt++] = 0xAA;
	Data_to_Send[_cnt++] = 0xAA;
	Data_to_Send[_cnt++] = 0xF1;
	Data_to_Send[_cnt++] = 14;
	
	Data_to_Send[_cnt++] = BYTE1(A);
	Data_to_Send[_cnt++] = BYTE0(A);
	Data_to_Send[_cnt++] = BYTE1(B);
	Data_to_Send[_cnt++] = BYTE0(B);
	Data_to_Send[_cnt++] = BYTE1(C);
	Data_to_Send[_cnt++] = BYTE0(C);
	Data_to_Send[_cnt++] = BYTE1(D);
	Data_to_Send[_cnt++] = BYTE0(D);
	Data_to_Send[_cnt++] = BYTE1(E);
	Data_to_Send[_cnt++] = BYTE0(E);	
	Data_to_Send[_cnt++] = BYTE1(F);
	Data_to_Send[_cnt++] = BYTE0(F);	
	Data_to_Send[_cnt++] = BYTE1(G);
	Data_to_Send[_cnt++] = BYTE0(G);	
	
	for(i = 0; i < _cnt; i++)
	{
		sumcheck += Data_to_Send[i];
		addcheck += sumcheck;
	}
	
	Data_to_Send[_cnt++]=sumcheck;
	Data_to_Send[_cnt++]=addcheck;
	HAL_UART_Transmit_IT(&huart1, Data_to_Send,_cnt);
}
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* USER CODE BEGIN 1 */
	int num = 0;
	int num1 = 0;
	u8 key = 0;
	u8 Concentration[12];
	uint16_t AD_Value[7];
	uint16_t AD_Value_baseline[7];

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */
	delay_init(168);
  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_ADC1_Init();
  MX_FSMC_Init();
  MX_USART1_UART_Init();
  MX_CRC_Init();
  MX_X_CUBE_AI_Init();
  /* USER CODE BEGIN 2 */
  //KEY_Init();
	aiInit();
  HAL_ADC_Start_DMA(&hadc1,(uint32_t*)&AD_Value,7);
	LCD_Init();
	LCD_Clear(WHITE);
	POINT_COLOR=RED;
	HAL_Delay(1000);
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */
    /* USER CODE BEGIN 3 */
	if(num==360){
		num = 0;
		acquire_and_process_data(in_data);
		aiRun(in_data, out_data);
		post_process(out_data);
		} 
	num++;
	//HAL_GPIO_TogglePin(GPIOF,GPIO_PIN_9);
	//HAL_GPIO_TogglePin(GPIOF,GPIO_PIN_10);

	sprintf((char*)Concentration,"%.1f(ppm)",potency1);
	POINT_COLOR=RED;	  
	LCD_ShowString(280,40,240,32,32,"Electronic Nose");
	POINT_COLOR=BLACK;
	LCD_ShowString(100,100,210,32,32,"Sensor-1:");		
	LCD_ShowString(100,140,210,32,32,"Sensor-2:");	
	LCD_ShowString(100,180,210,32,32,"Sensor-3:");	
	LCD_ShowString(100,220,210,32,32,"Sensor-4:");	
	LCD_ShowString(100,260,210,32,32,"Sensor-5:");	
	LCD_ShowString(100,300,210,32,32,"Sensor-6:");	
	LCD_ShowString(100,340,210,32,32,"Sensor-7:");			
	LCD_ShowxNum(260,100,AD_Value[0],4,32,0);  
	LCD_ShowxNum(260,140,AD_Value[1],4,32,0); 
	LCD_ShowxNum(260,180,AD_Value[2],4,32,0); 
	LCD_ShowxNum(260,220,AD_Value[3],4,32,0); 
	LCD_ShowxNum(260,260,AD_Value[4],4,32,0); 
	LCD_ShowxNum(260,300,AD_Value[5],4,32,0); 
	LCD_ShowxNum(260,340,AD_Value[6],4,32,0); 			
	LCD_ShowString(460,220,350,32,32,"Gas category:Ethylene");
	LCD_ShowString(460,300,350,32,32,"Current concentration:");	
	LCD_ShowString(560,340,200,32,32,Concentration);	
	POINT_COLOR=RED;
	LCD_ShowString(280,440,210,32,32,"Made by SWU_AI");
	AnoTc_SendUserTest(AD_Value[0],AD_Value[1],AD_Value[2],AD_Value[3],AD_Value[4],AD_Value[5],AD_Value[6]);
//	AD_Value[0] = 2146;
//	AD_Value[1] = 1749;
//	AD_Value[2] = 2369;
//	AD_Value[3] = 773;
//	AD_Value[4] = 2349;
//	AD_Value[5] = 1352;
//	AD_Value[6] = 2247;

	if (num1 == 360) num1 = 0;
	if (num1 == 0)
	{
		for (int i = 0; i < 7; i++) {
		AD_Value_baseline[i] = AD_Value[i];
	}
	}
	for (int i = 0; i < 7; i++) {
		in_buf[num1 * 7 + i] = AD_Value[i]-AD_Value_baseline[i]/AD_Value_baseline[i];
	}
	num1++;
	HAL_Delay(1000);
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 168;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
