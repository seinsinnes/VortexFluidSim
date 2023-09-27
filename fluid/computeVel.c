#include <math.h>
#include <float.h>
#include <stdio.h>

#define FourPi (4.0f * 3.1415926535897932384626433832795f)
const float  OneOverFourPi       = 1.0f / FourPi;
#define  sAvoidSingularity    pow(FLT_MIN, 1.0f / 3.0f)

extern void test(double *a);
extern void accumulateVelocity(double *vVelocity, double* vPosQuery , double* vortPos, double* vortVort, double vortRadius );

void subtractVectors3D( double *result, double *a, double *b )
{
	result[0] = a[0] - b[0];
	result[1] = a[1] - b[1];
	result[2] = a[2] - b[2];
}

void addVectors3D(double *result, double *a, double *b)
{
	result[0] = a[0] + b[0];
	result[1] = a[1] + b[1];
	result[2] = a[2] + b[2];
}

double dist2_3D( double *a)
{
	return a[0]*a[0] + a[1]*a[1] + a[2]*a[2]; 
}

void crossProduct(double *result, double *a, double *b)
{
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
}

void scalarProduct(double *result, double *a, double b)
{
	result[0] = a[0]*b;
	result[1] = a[1]*b;
	result[2] = a[2]*b;

}



void accumulateVelocity(double *vVelocity, double* vPosQuery , double* vortPos, double* vortVort, double vortRadius )
{
        //VortonInfo = [positon, vorticity, radius]                                                                                                  
        double vNeighborToSelf[3];
	double velocityTemp[3];
	double dist2;
	double oneOverDist;
	double distLaw;
	double radius2;

	double vcProdVortNeighbour[3];
	
	//printf(" %f \n", vortPos[0]);
	subtractVectors3D(vNeighborToSelf, vPosQuery, vortPos);
	//vNeighborToSelf     = vPosQuery - vortonInfo[0]                                       
      	radius2             = vortRadius * vortRadius;
        dist2               = dist2_3D(vNeighborToSelf) + sAvoidSingularity;
	              
        oneOverDist         = 1.0f / sqrt( dist2 );
	 
        //vNeighborToSelfDir  = vNeighborToSelf * oneOverDist
        /*If the reciprocal law is used everywhere then when 2 vortices get close, they tend to jettison. '''
        '''Mitigate this by using a linear law when 2 vortices get close to each other.*/
        //print vVelocity
        if (dist2 < radius2)
	{
            // Inside vortex core
            distLaw = ( oneOverDist / radius2 );
	}        
	else
	{
            //Outside vortex core                             
            distLaw = ( oneOverDist / dist2 );
	}
	//printf("%f\n",OneOverFourPi);
        //vVelocity +=  OneOverFourPi * ( 8.0 * radius2 * mRadius ) * numpy.cross(mVorticity, vNeighborToSelf) * distLaw
	crossProduct(vcProdVortNeighbour, vortVort, vNeighborToSelf);
	scalarProduct(velocityTemp, vcProdVortNeighbour, OneOverFourPi * ( 8.0f * radius2 * vortRadius )*distLaw);
        //vVelocity =  OneOverFourPi * ( 8.0 * radius2 * vortonInfo[2][0] ) * crossProduct(vortonInfo[1], vNeighborToSelf) * distLaw   
        //print vVelocity
	addVectors3D(vVelocity,vVelocity, velocityTemp);
}

void test(double *b)
{
	printf("%f %f %f\n",b[0],b[1],b[2]);
}
