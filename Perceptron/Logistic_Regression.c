/**
 * @file main.c
 * @mainpage Logistic Regression in C.
 * @brief Implementation of logistic Regression algorithm in C using pima-indians-diabetes dataset.
 * @author Waleed Ahmed Daud.
 * @Website waleed-daud.github.io
 * @Linkedin https://linkedin/in/waleed-daud-78472b9b
 * @Email waleed.daud@outlook.com
 * @date OCT 21
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define NUM_OF_TRAINING_OBSERVATIONS 600
#define NUM_OF_TESTING_OBSERVATIONS 168
#define NUM_OF_FEATURES  8
#define NUM_OF_ITERATIONS 4000
#define LEARNING_RATE 0.01


void delay(unsigned int mseconds)
{
    clock_t goal = mseconds + clock();
    while (goal > clock());
}


void Weights_init(double *weights)
{
    int j;
    for(j=0;j<NUM_OF_FEATURES;j++)

    {
        *(weights+j)=0;//(double)rand()/RAND_MAX;
    }
}

void reset_parameters(double  *z, double  *a, double *gradient)
{
    int j;
    for(j=0;j<NUM_OF_TRAINING_OBSERVATIONS;j++)
    {
        *(z+j)=0;
        *(a+j)=0;
        *(gradient+j)=0;
    }

}

double  sigmoid(double  z)
{
    if (z < -45.0) return 0;
    if (z > 45.0) return 1;
    double  a=(1/(1+exp(-z)));

    return a;
}


void export_weights(double *weights,double*bias,double *test_features_average)
{
    FILE *fp;
    fp=fopen("weights.txt","w");

    if(fp!=NULL)
    {
    int i;
    for(i=0;i<NUM_OF_FEATURES;i++)
    {
        fprintf(fp,"%lf,",weights[i]);
    }
    fprintf(fp,"\n bias: %lf \n",bias[0]);

    for(i=0;i<NUM_OF_FEATURES;i++)
    {
        fprintf(fp,"%lf,",test_features_average[i]);
    }
    fclose(fp);
    }

    else
    {printf(" File can't be opened !");}
}


int main()
{

double  train_data[NUM_OF_TRAINING_OBSERVATIONS][NUM_OF_FEATURES];
double  train_label[NUM_OF_TRAINING_OBSERVATIONS];             /// true values.
double predicted_Train_label [NUM_OF_TRAINING_OBSERVATIONS];

double  weights[NUM_OF_FEATURES];
double  bias[NUM_OF_TRAINING_OBSERVATIONS]={1.0};


double  test_data[NUM_OF_TESTING_OBSERVATIONS][NUM_OF_FEATURES];
double  test_label[NUM_OF_TESTING_OBSERVATIONS];             /// true test values.
double predicted_Test_label  [NUM_OF_TESTING_OBSERVATIONS];        /// predicted_Test_label  .


/// ########################################
double  z[NUM_OF_TRAINING_OBSERVATIONS]={0.0};
double  a[NUM_OF_TRAINING_OBSERVATIONS]={0.0};            /// predicted values.
double  gradient[NUM_OF_TRAINING_OBSERVATIONS];         /// gradient.

double train_features_average[NUM_OF_FEATURES];
double test_features_average[NUM_OF_FEATURES];

/// ################################################### Reading train_data #######################################################
int row,column=0;

FILE *fp;
fp=fopen("pima-indians-diabetes.txt","r");

if(fp!=NULL)
{
for(row=0;row<NUM_OF_TRAINING_OBSERVATIONS;row++)
{
    fscanf(fp,"%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n",&train_data[row][column],&train_data[row][column+1],&train_data[row][column+2],&train_data[row][column+3],&train_data[row][column+4],&train_data[row][column+5],
           &train_data[row][column+6],&train_data[row][column+7],&train_label[row]);

}
fclose(fp);
}
else
{
printf("File Can not be opened !");

}


/// ################################################## Reading Test train_data #######################################################
fp=fopen("pima-indians-diabetes_test.txt","r");
column=0;
 if(fp!=NULL)
{
    for(row=0;row<NUM_OF_TESTING_OBSERVATIONS;row++)
{
            fscanf(fp,"%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n",&test_data[row][column],&test_data[row][column+1],&test_data[row][column+2],&test_data[row][column+3],&test_data[row][column+4],&test_data[row][column+5],&test_data[row][column+6],
                   &test_data[row][column+7],&test_label[row]);
}
fclose(fp);
}

else
{
printf("File Can not be opened !");

}

///################################################### Features normalization/Training #################################################

int j,i;


for(i=0;i<NUM_OF_FEATURES;i++)
{
    double average=0;

    for(j=0;j<NUM_OF_TRAINING_OBSERVATIONS;j++)

    {
        average+=train_data[j][i];
    }

    average=(double)(average/NUM_OF_TRAINING_OBSERVATIONS);
    train_features_average[i]=average;

    //printf("average:feature: %d -----------> %lf \n",i,average);

}


for(i=0;i<NUM_OF_FEATURES;i++)
{

    for(j=0;j<NUM_OF_TRAINING_OBSERVATIONS;j++)
    {

       // printf(" train_data before: %lf \n ",train_data[i][j]);
        train_data[j][i]=((train_data[j][i]-train_features_average[i])/NUM_OF_TRAINING_OBSERVATIONS);
       // printf(" train_data after: %lf \n ",train_data[i][j]);

    }

}

///################################################### Features normalization/Testing #################################################


for(i=0;i<NUM_OF_FEATURES;i++)
{
    double average=0;

    for(j=0;j<NUM_OF_TESTING_OBSERVATIONS;j++)

    {
        average+=test_data[j][i];
    }

    average=(double)(average/NUM_OF_TESTING_OBSERVATIONS);
    test_features_average[i]=average;

    //printf("average:feature: %d -----------> %lf \n",i,average);

}


for(i=0;i<NUM_OF_FEATURES;i++)
{

    for(j=0;j<NUM_OF_TESTING_OBSERVATIONS;j++)
    {

       // printf(" train_data before: %lf \n ",train_data[i][j]);
        test_data[j][i]=((test_data[j][i]-test_features_average[i])/NUM_OF_TESTING_OBSERVATIONS);
       // printf(" train_data after: %lf \n ",train_data[i][j]);

    }

}

/// ################################################## algorithm ##############################################################

Weights_init(weights);                                                   /// initialize weights

/// ########################################### Iterations ######################################################################
FILE *cost_report_fp;
cost_report_fp = fopen("Cost_report.txt", "a");

int iteration;
double  cost=0;
for(iteration=0;iteration<NUM_OF_ITERATIONS;iteration++)
{
cost=0;
reset_parameters(z,a,gradient);

/// ########################################### Forward Propagation ###############################################################
int i=0,j=0;

        for(i=0;i<NUM_OF_TRAINING_OBSERVATIONS;i++)
        {
             for(j=0;j<NUM_OF_FEATURES;j++)
        {
            z[i]+=weights[j]*train_data[i][j];

        }
        z[i]+=bias[i];

        a[i]=sigmoid(z[i]);

        }

/// #############################################################################################################################
/// ############################################ Cost Function ##################################################################
/// #############################################################################################################################


for(j=0;j<NUM_OF_TRAINING_OBSERVATIONS;j++)

    {//printf(" %lf \t \n",log(1-a[j]));
        cost+=-(train_label[j]*log(a[j])+(1-train_label[j])*log(1-a[j]));
        //cost+=pow((a[j]-train_label[j]),2)/2;
    }

    //printf("COST::::: %d \t \n",cost);

cost=(double)cost/NUM_OF_TRAINING_OBSERVATIONS;

if(iteration%1000==0) {printf("iteration:%d --------------> cost: %lf \n",iteration,cost);}

fprintf(cost_report_fp,"iteration:%d --------------> cost: %lf \n",iteration,cost);
/// ######################################### gradient #########################################################################

//delay(1000);
for(j=0;j<NUM_OF_TRAINING_OBSERVATIONS;j++)
{

 gradient[j]=a[j]-train_label[j];

//printf(" a: %lf ----------------------------> train_label: %lf \n",a[j],train_label[j]);

}


//printf("###################################### NOW WE WILL UPDATE THE WEIGHTS ! #######################################");
/// ######################################### Update Weights and bias ##################################################################

for(i=0;i<NUM_OF_FEATURES;i++)
{
    double  results[NUM_OF_TRAINING_OBSERVATIONS]={0}, sum=0, gradient_sum=0;//sum2[NUM_OF_TRAINING_OBSERVATIONS]={0};
    for(j=0;j<NUM_OF_TRAINING_OBSERVATIONS;j++)
    {
       results[j]=gradient[j]*train_data[j][i];
      // printf(" GRADIENT: %lf \t \n",gradient[i]);
        gradient_sum+=gradient[j];
    }

    for(j=0;j<NUM_OF_TRAINING_OBSERVATIONS;j++)
    {
       sum+=results[j];
    }


   // printf(" delta: %lf \t \n",(double)(LEARNING_RATE*sum/NUM_OF_TRAINING_OBSERVATIONS));

     weights[i]-=(double)(LEARNING_RATE*sum/NUM_OF_TRAINING_OBSERVATIONS);           /// update the weights

    for(j=0;j<NUM_OF_TRAINING_OBSERVATIONS;j++)
    {
       bias[j]-=(double)(LEARNING_RATE*gradient_sum/NUM_OF_TRAINING_OBSERVATIONS);   /// update the bias
    }


}




}

/// ################################################### Forward propagation for Testing #########################################

reset_parameters(z,a,gradient);
        for(i=0;i<NUM_OF_TESTING_OBSERVATIONS;i++)
        {
             for(j=0;j<NUM_OF_FEATURES;j++)
        {
            z[i]+=weights[j]*test_data[i][j];

        }
        z[i]+=bias[i];

        predicted_Test_label[i]=sigmoid(z[i]);

        }

/// ################################################## Testing Accuracy #########################################################
    double counter=0;
    for(i=0;i<NUM_OF_TESTING_OBSERVATIONS;i++)
    {

    if(predicted_Test_label  [i]>0.5) predicted_Test_label[i]=1;
    else if(predicted_Test_label  [i]<0.5) predicted_Test_label[i]=0;

    if(predicted_Test_label[i]==test_label[i])counter++;
    printf("actual: [%lf] ,predicted: [%lf].\n",test_label[i],predicted_Test_label[i]);

    }

    double accuracy=(counter/NUM_OF_TESTING_OBSERVATIONS)*100 ;

    printf("\n\n\n  Test Accuracy is: %lf ",accuracy);

/// ################################################## export weights #####################################################
    export_weights(weights,bias,test_features_average);
/// ################################################## closing files ######################################################
fclose(cost_report_fp);


/// #########################################################################################################################

    return 0;
}
