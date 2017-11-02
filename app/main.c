/**
  ******************************************************************************
  * @file    GPIO/GPIO_IOToggle/Src/main.c
  * @author  MCD Application Team
  * @version V1.2.0
  * @date    05-Dec-2014
  * @brief   This example describes how to configure and use GPIOs through
  *          the STM32F0xx HAL API.
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; COPYRIGHT(c) 2014 STMicroelectronics</center></h2>
  *
  * Redistribution and use in source and binary forms, with or without modification,
  * are permitted provided that the following conditions are met:
  *   1. Redistributions of source code must retain the above copyright notice,
  *      this list of conditions and the following disclaimer.
  *   2. Redistributions in binary form must reproduce the above copyright notice,
  *      this list of conditions and the following disclaimer in the documentation
  *      and/or other materials provided with the distribution.
  *   3. Neither the name of STMicroelectronics nor the names of its contributors
  *      may be used to endorse or promote products derived from this software
  *      without specific prior written permission.
  *
  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  *
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include  <math.h>

#define NUM_OF_FEATURES  8
#define NUM_OF_TESTING_OBSERVATIONS 168

/** @addtogroup STM32F0xx_HAL_Examples
  * @{
  */

/** @addtogroup GPIO_IOToggle
  * @{
  */

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
static GPIO_InitTypeDef  GPIO_InitStruct;

/* Private function prototypes -----------------------------------------------*/
static void SystemClock_Config(void);
static void Error_Handler(void);

/* Private functions ---------------------------------------------------------*/

/**
  * @brief  Main program
  * @param  None
  * @retval None
  */

double exp(double n)
{
	double result=1;
	if(n>0)
	{
	int i;
	for(i=0;i<n;i++)
	{
		result*=2.71828182846;
	}
	}

	else if(n<0)
	{
		n=n*-1;
		int i;
			for(i=0;i<n;i++)
			{
				result*=2.71828182846;
			}
			result=1/result;
	}
	return result;
}
double  sigmoid(double  z)
{
    if (z < -45.0) return 0;
    if (z > 45.0) return 1;
    double  a=(1/(1+exp(-z)));

    return a;
}

int main(void)
{
  /* This sample code shows how to use GPIO HAL API to toggle LED3 and LED4 IOs
    in an infinite loop. */

  /* STM32F0xx HAL library initialization:
       - Configure the Flash prefetch
       - Systick timer is configured by default as source of time base, but user 
         can eventually implement his proper time base source (a general purpose 
         timer for example or other time source), keeping in mind that Time base 
         duration should be kept 1ms since PPP_TIMEOUT_VALUEs are defined and 
         handled in milliseconds basis.
       - Low Level Initialization
     */
  HAL_Init();

  /* Configure the system clock to 48 MHz */
  SystemClock_Config();
  
  /* -1- Enable each GPIO Clock (to be able to program the configuration registers) */
  LED3_GPIO_CLK_ENABLE();
  LED4_GPIO_CLK_ENABLE();

  /* -2- Configure IOs in output push-pull mode to drive external LEDs */
  GPIO_InitStruct.Mode  = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull  = GPIO_PULLUP;
  GPIO_InitStruct.Speed = GPIO_SPEED_HIGH;

  GPIO_InitStruct.Pin = LED3_PIN;
  HAL_GPIO_Init(LED3_GPIO_PORT, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = LED4_PIN;
  HAL_GPIO_Init(LED4_GPIO_PORT, &GPIO_InitStruct);

  /* -3- Toggle IOs in an infinite loop */
  double test_data[NUM_OF_FEATURES]={7,168,88,42,321,38.2,0.787,40};
  double weights[NUM_OF_FEATURES]={0.022117,0.450796,0.036391,0.032870,0.448000,0.078911,0.001877,0.077874};
  double bias=0.363157;
  double test_features_average[NUM_OF_FEATURES]={3.934524,123.607143,70.619048,20.458333,80.767857,32.304167,0.439065,33.107143};

///##################### Features normalization ################################
int i;
  for(i=0;i<NUM_OF_FEATURES;i++)
  {

  test_data[i]=((test_data[i]-test_features_average[i])/NUM_OF_TESTING_OBSERVATIONS);

  }

/// ############################################################################

  double z=0;
  int j;
  for(j=0;j<NUM_OF_FEATURES;j++)
    {
       z+=weights[j]*test_data[j];
    }
  z+=bias;

  double predicted_value=sigmoid(z);
  if(predicted_value>0.5)predicted_value=1;
  else if(predicted_value<0.5)predicted_value=0;

  while (1)
  {

	if(predicted_value==1.0)
	{
    HAL_GPIO_TogglePin(LED3_GPIO_PORT, LED3_PIN);
    /* Insert delay 100 ms */
    HAL_Delay(100);
	}

	else if(predicted_value==0.0)
	{
    HAL_GPIO_TogglePin(LED4_GPIO_PORT, LED4_PIN);
    /* Insert delay 100 ms */
    HAL_Delay(100);
	}
  }

}

/**
  * @brief  System Clock Configuration
  *         The system Clock is configured as follow : 
  *            System Clock source            = PLL (HSE)
  *            SYSCLK(Hz)                     = 48000000
  *            HCLK(Hz)                       = 48000000
  *            AHB Prescaler                  = 1
  *            APB1 Prescaler                 = 1
  *            HSE Frequency(Hz)              = 8000000
  *            PREDIV                         = 1
  *            PLLMUL                         = 6
  *            Flash Latency(WS)              = 1
  * @param  None
  * @retval None
  */
static void SystemClock_Config(void)
{
  RCC_ClkInitTypeDef RCC_ClkInitStruct;
  RCC_OscInitTypeDef RCC_OscInitStruct;
  
  /* Select HSE Oscillator as PLL source */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PREDIV = RCC_PREDIV_DIV1;
  RCC_OscInitStruct.PLL.PLLMUL = RCC_PLL_MUL6;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct)!= HAL_OK)
  {
    Error_Handler();
  }

  /* Select PLL as system clock source and configure the HCLK and PCLK1 clocks dividers */
  RCC_ClkInitStruct.ClockType = (RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_PCLK1);
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_1)!= HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief  This function is executed in case of error occurrence.
  * @param  None
  * @retval None
  */
static void Error_Handler(void)
{
  /* User may add here some code to deal with this error */
  while(1)
  {
  }
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
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */

  /* Infinite loop */
  while (1)
  {
  }
}
#endif

/**
  * @}
  */

/**
  * @}
  */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
