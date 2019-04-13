#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "pbc.h"
#include <string.h>

extern double yrandom(double, double);
long random(void);


////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////Molecular-Dynamics-Verlet-scheme///////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////

int md_integrator_verlet( int N , double (*r)[3] , double (*v)[3] , double *q , double DT , int N_steps ,
                          double a ,double  b ,double  c ,double  r0 ,
                          double  alpha , double  rcut , double  kcut ,
                          int use_brute_force , int use_minimg , char hard_wall[3])
{
  double (*force1)[3] , (*force2)[3] , Epot , Ekin , *mass ;
  int jj , ii, tt, use_gpu = 0 ;
  double e1 = 1.6e-19 , Coulomb = 9e9 , r_norm = 1e-10 ;
  char on[2] , off[3];
  strcpy (on, "ON");strcpy (off, "OFF");
  force1 = calloc( N * 3 , sizeof(double) ) ;
  force2 = calloc( N * 3 , sizeof(double) ) ;
  v      = calloc( N * 3 , sizeof(double) ) ;
  mass   = calloc( N , sizeof(double) ) ;

  for ( ii = 0 ; ii < N ; ii++ ) 
  { 
      mass[ii] = 9.1e-31 ;
      for (jj=0;jj<3;++jj) { v[ii][jj] = 6000000 ;  r[ii][jj] *= r_norm ; } 
  }

  FILE *fp = fopen( "./pbc_energy.txt" , "w" ) ; 
  for ( tt =0 ; tt < N_steps ; tt++ )
  {

    periodicbc( N , q , r , a * r_norm , b * r_norm , c * r_norm ,
                force1 , NULL , NULL , 
                r0 * r_norm , alpha / r_norm , rcut , kcut ,
                use_brute_force , use_gpu , use_minimg ) ;

    for ( ii = 0 ; ii < N ; ++ii ) {   for ( jj = 0 ; jj < 3 ; ++jj ) {
        r[ii][jj] += v[ii][jj] * DT + 0.5 * (Coulomb * e1 * e1) * force1[ii][jj] / mass[ii] * DT*DT ;
    }}

    apply_3Dperiodic_box( a * r_norm , b * r_norm , c * r_norm , N , r , NULL ) ; 

    periodicbc( N , q , r , a * r_norm , b * r_norm, c * r_norm ,
                force2 , NULL , NULL , 
                r0 * r_norm , alpha / r_norm , rcut , kcut ,
                use_brute_force , use_gpu , use_minimg ) ;
 
    for ( ii = 0 ; ii < N ; ++ii ) {   for ( jj = 0 ;jj < 3 ; ++jj ) {
        v[ii][jj] += 0.5 * (Coulomb * e1 * e1) * ( force1[ii][jj] + force2[ii][jj] ) / mass[ii] * DT ;
    }}

   double centre[3] = {0.,0.,0.} , abc[3] = {a*r_norm ,b*r_norm ,c*r_norm} ;
   if (compare_string(hard_wall, on) == 0)
     {
      apply_hard_wall_box( r , v , N , a * r_norm * 0.9 , centre , abc ) ;
     }

    Ekin = 0 ; for ( ii = 0 ; ii < N ; ii++ ) { for (jj=0;jj<3;++jj) { Ekin += 0.5 * mass[ii] * v[ii][jj] * v[ii][jj] ; } }
    periodicbc( N , q , r , a * r_norm , b * r_norm , c * r_norm ,
                NULL , NULL , &Epot , 
                r0 * r_norm , alpha / r_norm , rcut , kcut ,
                use_brute_force , use_gpu , use_minimg ) ;
    Epot *= (Coulomb * e1 * e1) ;
    fprintf( fp , "%le\n" , ( Ekin + Epot ) / e1 ) ;

  }

  fclose(fp) ;

  return 0 ;
}


///////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////PERIODIC SWAPPING ROUTINE/////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

int apply_3Dperiodic_box( double a , double b , double c , int NN , double (*r1)[3] , double (*r2)[3] ) 
{
    double abc[3] = { a , b , c } ;
    double shift ;
    int cc , pp ;
    int indic = 1 ;

    for ( cc = 0 ; cc < 3 ; cc++ )
    {
        if ( abc[cc] > 0 ) 
        {
            for ( pp=0 ; pp < NN ; pp++ )
            {
                shift = floor( ( r1[pp][cc] + abc[cc] / 2.0 ) / abc[cc] ) ;
                if ( (indic) && (0!=shift) ) { indic = 0  ; }
                r1[pp][cc] -= shift * abc[cc] ;
                if ( NULL != r2 ) {   r2[pp][cc] -= shift * abc[cc] ;   }
                //if ( (indic) && (0!=shift) ) { indic = 0  ; printf("Shift %d\n", (int)(shift) ); }
                //if ( (indic) && (0!=shift) ) { printf("Shift %d\n", (int)(shift) ); indic = 0 ;} // Original
            }
        }
    }

    return 0 ;
}

/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////HARDWALL ROTUINE////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////

#if ! defined SIGN 
#define SIGN(x) ( (x < 0) ? -1 : (x > 0) )
#endif

int apply_hard_wall_box( double (*a_r)[3] , double (*a_v)[3] , int a_N , 
                         double a_D , double *a_center , double *a_per3D ) 
{

   if ( 0 == a_D ) { return 0 ; }

   double per3D_abc[3] = { a_per3D[0] , a_per3D[1] , a_per3D[2] } ;
   int is_per = (int) SIGN( per3D_abc[0] * per3D_abc[1] * per3D_abc[2] ) ;
   // To avoid division by 0 :
   if ( ! is_per ) { per3D_abc[0] = per3D_abc[1] = per3D_abc[2] = 1 ; } 

   double zero_vect[3] = { 0 , 0 , 0 } ;
   double pos[3] , halfD ;
   int ii , kk ;

   //* If edge has zero length don't do anything */
   if ( 0 == a_D ) { return 0 ; }

   halfD = a_D / 2.0 ;

   if ( NULL == a_center ) { a_center = zero_vect ; }

   for ( ii = 0 ; ii < a_N ; ii++ )
   {
       // shift according to center of box
       for ( kk = 0 ; kk < 3 ; kk++ ) { pos[kk] = a_r[ii][kk] - a_center[kk] ; }
       for ( kk = 0 ; kk < 3 ; kk++ )
       {
       // shift according to the box the particle is in
           pos[kk] -= is_per * floor( ( pos[kk] + per3D_abc[kk]/2.0 ) / per3D_abc[kk] ) * per3D_abc[kk] ;
           if ( halfD < fabs( pos[kk] ) )
           {
       // bouncing back if needed
               a_v[ii][kk] *= - ( SIGN( pos[kk] ) * SIGN( a_v[ii][kk] ) ) ;
           }
       }
   }

   return 0 ;
}

/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// String Compare Routine///////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////




int compare_string(char *first, char *second) {
   while (*first == *second) {
      if (*first == '\0' || *second == '\0')
         break;
 
      first++;
      second++;
   }
 
   if (*first == '\0' && *second == '\0')
      return 0;
   else
      return -1;
}

