#include "math_util.h"

// float rand_normal(float mu, float sigma){
//   printf("%f \n", M_PI);
//   printf("%f \n", Uniform());
//   printf("%f \n", log(Uniform()));
//   printf("%f \n", sin(2.0*M_PI*Uniform()));
//   printf("sqrt %f \n", sqrt(2.0f));
//   printf("cal sqrt %f \n", sqrt(-2.0*log(Uniform())));
//   double z=sqrt(-2.0*log(Uniform())) * sin(2.0*M_PI*Uniform());
//   printf("%f \n", z);
//   double out = mu + sigma*z;
//   printf("%f\n", out);
//   return (float)out;
// }

float rand_normal()
{
    static double rand1, rand2;

    rand1 = rand() / ((double) RAND_MAX);
    if(rand1 < 1e-100) rand1 = 1e-100;
    rand1 = -2 * log(rand1);
    rand2 = (rand() / ((double) RAND_MAX)) * TWO_PI;

    return sqrt(rand1) * cos(rand2);
}
