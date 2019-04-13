

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <omp.h>

#ifdef USE_CUDA
#include "./lib/cuda_nbody.h"
#endif

static double diff_time(struct timespec start,struct timespec end) {
  return( (end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1000000000 );
}


int periodicbc(int N , double *q , double (*cq)[3] , double a , double b , double c ,double (*cpyaq)[3] , double *potential , double *tpot , double rp ,double alpha , int rcut , int kcut , int select , int gpu , int min_img) 
 {

              /********************************************************/
              /********************************************************/
              /*************Call for CPU & GPU Ewald calc**************/
              /*****************and for Brute Force********************/
              /********************************************************/   

   int sx = 0 ;
     if (NULL == potential)
      {
	sx = 1 ;
	potential = calloc( N , sizeof(double) ) ;
      }

     if (1 == gpu)
      {
        omp_set_num_threads(1) ;
//      printf( "Num of threads: %d \n" , omp_get_max_threads() ) ;
        ewaldperiodicbc(q , cq , cpyaq , potential , tpot , alpha , N , a , b , c , rcut , kcut , gpu , rp , min_img);
      }
  
    
     if (1 == select  )
      {
        omp_set_num_threads(1) ;
//      printf( "Num of threads: %d \n" , omp_get_max_threads() ) ;
        bforcepbc(q , cq , cpyaq , potential , tpot  , N , a , b , c , rcut , rp , min_img);
      }
  
     if(1 == sx )
      {
        free(potential) ;
      }
  
 }


int ewaldperiodicbc(double *q , double (*cq)[3] , double (*cpyaq)[3] , double *potential , double *tpot , double alpha , int N , double a , double b , double c ,  int rcut , int kcut , int gpu , double rp , int min_img)
  {
    int i , j , nbody , n0 , n1 , n2 , n , r0 , r1 , r2 ;
    double  *u_self  , *u_real , (*F_k)[3] , (*F_r)[3] , (*F_d)[3];
    double  vect[3] , lattice[3] , dc , dd , dd2 , sqdd ;
    double  Volume , d , e , f , g , h , x , y , u_r , u_d , u_s  , u_k , u_n , k2 , dk , sum , Q , u_t , rpot ;
    double   sqk , ri ;
    double  params[8] ;
    double shift_coord0 , shift_coord1 , shift_coord2  ;   
    double rvect1 , rvect2 , rvect3 ;
    
    
    lattice[0] = a ;
    lattice[1] = b ;
    lattice[2] = c ;
    dc   = 1 ;
    nbody = N ;
    u_r = 0 ;
    u_s = 0 ;
    u_k = 0 ;
    u_t = 0 ;
    u_n = 0 ;
    u_d = 0 ;
    Volume = lattice[0] * lattice[1] * lattice[2] ;           
    u_self = calloc( nbody , sizeof(double) ) ;             
    u_real = calloc( nbody , sizeof(double) ) ;
    F_k = calloc( nbody * 3 , sizeof(double) ) ;
    F_r = calloc( nbody * 3 , sizeof(double) ) ;
    F_d = calloc( nbody * 3 , sizeof(double) ) ;
    
    
                         /**************************************************/   
    #if defined USE_CUDA    
    if ( 1 == gpu )
      {
         double (*Rforces)[3] ;
         double (*Kforces)[3] ;
  //       double (*Dforces)[3] ;
         Rforces = calloc( N * 3 , sizeof(double) ) ;
         Kforces = calloc( N * 3 , sizeof(double) ) ;
  //       Dforces = calloc( N * 3 , sizeof(double) ) ;
         double gpotential ;
         int ii = 0 ;
         time_t start,end;
  
         params[0] = rp * rp ;
         params[1] = a ;
         params[2] = b ;
         params[3] = c ;
         params[4] = rcut ;
         params[5] = kcut ;
         params[6] = alpha ;
         params[7] = min_img ;

  
         if (NULL != tpot)
            {
             *tpot = 0 ;
                 if(0 == alpha)
                 { 
   		 ////////////////////////////////////////////////////////////////////  
		 ///////////////Real-space potential using GPU///////////////////////  
		 ////////////////////////////////////////////////////////////////////  
                 struct timespec tht1,tht2;
                 clock_gettime(CLOCK_REALTIME, &tht1);
                 cuda_ewald3d_pot_Rspace(cq , q , cq , q , params , &gpotential , nbody , nbody) ;    
                 clock_gettime(CLOCK_REALTIME, &tht2);

                 *tpot = gpotential;
                 }
                 else        
                 {
   		 ////////////////////////////////////////////////////////////////////  
		 ///////////////Real-space potential using GPU///////////////////////  
		 ////////////////////////////////////////////////////////////////////  

                  struct timespec tht1,tht2;
                  clock_gettime(CLOCK_REALTIME, &tht1);
                  cuda_ewald3d_pot_Rspace(cq , q , cq , q , params , &gpotential , nbody , nbody) ;    
                  clock_gettime(CLOCK_REALTIME, &tht2);
//                fprintf(stderr,"R-Space Potential %g\n",diff_time(tht1,tht2));

   		 ////////////////////////////////////////////////////////////////////  
		 /////////////////K-space potential using GPU////////////////////////  
		 ////////////////////////////////////////////////////////////////////  
   

                 r0 = r1 = r2 = kcut ;
                 u_k = 0 ; 
                 struct timespec tht3,tht4;
                 clock_gettime(CLOCK_REALTIME, &tht3);
              //   #pragma omp parallel for private( f , g ,rvect1,rvect2,d,rvect3,k2,sqk ,n0 , n1 , n2 , i)
                 for(n0 = -r0 ; n0 < r0 + 1 ; n0++)
                  {
                  for(n1 = -r1 ; n1 < r1 + 1 ; n1++)
                   {
                   for(n2 = -r2 ; n2 < r2 + 1 ; n2++)
                    {
                        e = f = g = 0 ;  
                	rvect1 = ( 2.0 * M_PI * n0 ) / lattice[0] ;
                	rvect2 = ( 2.0 * M_PI * n1 ) / lattice[1] ;
                	rvect3 = ( 2.0 * M_PI * n2 ) / lattice[2] ;
                	k2 = rvect1 * rvect1 + rvect2 * rvect2 + rvect3 * rvect3 ;
                	sqk = sqrt(k2);
                	if ( k2 != 0 )
	                 {
	                  d = exp(-k2 / (4.0 * alpha * alpha)) / k2;
	                  for (i = 0 ; i < nbody ; i++)
	                   { 
	                    f += q[i] * cos(rvect1 * cq[i][0] + rvect2 * cq[i][1] + rvect3 * cq[i][2]) ;
	                    g += q[i] * sin(rvect1 * cq[i][0] + rvect2 * cq[i][1] + rvect3 * cq[i][2]) ;
	                   } /*end for i loop*/
 	                   u_k += (( 2 * M_PI ) /  Volume ) * d * (( f * f ) + ( g * g )) ;
	         	  }  //endif 
  	                else
  	                 {
  	                  n0 = n1 = n2 = kcut ;
 	                 }
 	            } /*end for n2 loop*/
                   } /*end for n1 loop*/
                  }/*end for n0 loop*/
                 clock_gettime(CLOCK_REALTIME, &tht4);
//               fprintf(stderr,"K-Space Potential %g\n",diff_time(tht3,tht4));

   		 ////////////////////////////////////////////////////////////////////  
		 //////////////Self Interaction potential using GPU//////////////////  
		 ////////////////////////////////////////////////////////////////////  


                 i = dd = sqdd = Q = 0 ;
                 int my_id ;
                 double *dummy_array ;
                 dummy_array = calloc ( 6  , sizeof (double));
                 
                 #pragma omp parallel private(my_id,i)
                   {
                       my_id = omp_get_thread_num() ;
                     for (i = my_id ; i < nbody ; i += 6)
                       {
	                 dummy_array[my_id] += q[i] * q[i] ;
                       }
                   }
               
                     for (i = 0 ; i < 6 ; i++)
                       {
                         Q += dummy_array[i];
                       }
	                 u_s = -(alpha /sqrt(M_PI)) * Q ;
                         free (dummy_array);
   		 
                 
                 
                 ////////////////////////////////////////////////////////////////////  
		 /////////////////Dipole potential using GPU////////////////////////  
		 ////////////////////////////////////////////////////////////////////  
                 double  vect0 = 0 , vect1 = 0 , vect2 = 0 ;
                // #pragma omp parallel for private(shift_coord0,shift_coord1,shift_coord2,vect0,vect1,vect2,dd,i,j) 
                 for (i = 0 ; i < nbody ; i++)
                  {
                  for (j = 0 ; j < nbody ; j++)
	           {
	            shift_coord0 = cq[j][0] - lattice[0] * min_img * floor((( cq[j][0] - cq[i][0] ) /lattice[0] ) + (1.0/2.0)) ;
	            shift_coord1 = cq[j][1] - lattice[1] * min_img * floor((( cq[j][1] - cq[i][1] ) /lattice[1] ) + (1.0/2.0)) ;
                    shift_coord2 = cq[j][2] - lattice[2] * min_img * floor((( cq[j][2] - cq[i][2] ) /lattice[2] ) + (1.0/2.0));
               
	            vect0 = cq[i][0] * shift_coord0 ;
	            vect1 = cq[i][1] * shift_coord1 ;
	            vect2 = cq[i][2] * shift_coord2 ;
           
	            dd = q[i] * q[j] * ( vect0 + vect1 + vect2 ) ;
                    u_d += ((2 * M_PI) / ( 3 * Volume ) ) * dd ;
                   }  
                  } 
                 
                 ////////////////////////////////////////////////////////////////////  
		 ////////////////////Summing up the Potential////////////////////////  
		 ////////////////////////////////////////////////////////////////////
               //  printf("dipole-term % le\n",u_d);
                  *tpot = gpotential + u_s + u_k*2 + u_d ;
            //     printf("r_pot % le\n",gpotential);
           //        printf("k_pot % le\n",u_k*2);
                 // printf("s_pot GPU% le\n",u_s);
                //  printf("d_pot GPU % le\n",u_d);
                 }

           } 
 
		 
     if (NULL != cpyaq)
       {
         if(0 == alpha)
          {
                 ////////////////////////////////////////////////////////////////////  
		 ///////////////////Real-space Forces using GPU//////////////////////  
		 ////////////////////////////////////////////////////////////////////
                 struct timespec tht5,tht6;
                 clock_gettime(CLOCK_REALTIME, &tht5);
                 cuda_ewald3d_force_Rspace(cq , q , cq , q , params , Rforces , nbody , nbody) ;
                 clock_gettime(CLOCK_REALTIME, &tht6);

                 for (i = 0 ; i < nbody ; i++)
                  {
                  for (j = 0 ; j < 3 ; j++)
                   {
                   cpyaq[i][j] = Rforces[i][j];
                   }
                  } 
          }

         else
          {
                 ////////////////////////////////////////////////////////////////////  
		 ///////////////////Real-space Forces using GPU//////////////////////  
		 ////////////////////////////////////////////////////////////////////
                 struct timespec tht5,tht6;
                 clock_gettime(CLOCK_REALTIME, &tht5);
                 cuda_ewald3d_force_Rspace(cq , q , cq , q , params , Rforces , nbody , nbody) ;
                 clock_gettime(CLOCK_REALTIME, &tht6);
//               fprintf(stderr,"Real-Space Forces %g\n",diff_time(tht5,tht6));

                 ////////////////////////////////////////////////////////////////////  
		 //////////////////////K-space Forces using GPU//////////////////////  
		 ////////////////////////////////////////////////////////////////////

                 struct timespec tht7,tht8;
                 clock_gettime(CLOCK_REALTIME, &tht7);
                 cuda_ewald3d_force_Kspace(cq , q , cq , q , params , Kforces , nbody , nbody) ;   
                 clock_gettime(CLOCK_REALTIME, &tht8);
//                fprintf(stderr,"K-Space Forces %g\n",diff_time(tht7,tht8));
   
                 ////////////////////////////////////////////////////////////////////  
		 ///////////////Forces due to Surface Dipole term////////////////////  
		 ////////////////////////////////////////////////////////////////////
    
                 a = b = c  = sqdd = i = j = 0 ;


             //    #pragma omp parallel for private(shift_coord0,shift_coord1,shift_coord2,a,b,c,dd,i,j) 
                 for( i = 0 ; i < nbody ; i++ )
                  {
                   a = b = c = 0 ;
                   for( j = 0 ; j < nbody ; j++ )
                    {
	              shift_coord0 = cq[j][0] - lattice[0] * min_img * floor((( cq[j][0] - cq[i][0] ) /lattice[0] ) + (1.0/2.0)) ;
 	              shift_coord1 = cq[j][1] - lattice[1] * min_img * floor((( cq[j][1] - cq[i][1] ) /lattice[1] ) + (1.0/2.0)) ;
 	              shift_coord2 = cq[j][2] - lattice[2] * min_img * floor((( cq[j][2] - cq[i][2] ) /lattice[2] ) + (1.0/2.0)) ; 
	              a += q[j] *  shift_coord0 ;
	              b += q[j] *  shift_coord1 ;
	              c += q[j] *  shift_coord2 ;
                    }
                   F_d[i][0] =  ((4 * M_PI * q[i]) / ( 3 * Volume ) ) * a ;
                   F_d[i][1] =  ((4 * M_PI * q[i]) / ( 3 * Volume ) ) * b ;
                   F_d[i][2] =  ((4 * M_PI * q[i]) / ( 3 * Volume ) ) * c ;
                  }
               
          //       cuda_ewald3d_force_D(cq , q , cq , q , params , Dforces , nbody , nbody) ;
                 ////////////////////////////////////////////////////////////////////  
		 /////////////////////Summing up the Forces//////////////////////////  
		 ////////////////////////////////////////////////////////////////////

                 for (i = 0 ; i < nbody ; i++)
                  {
                  for (j = 0 ; j < 3 ; j++)
                   {
            //        cpyaq[i][j] = Dforces[i][j];
                    cpyaq[i][j] = Rforces[i][j] + Kforces[i][j] - F_d[i][j];
                   }
                  } 
          }
                 ////////////////////////////////////////////////////////////////////  
		 /////////////////////Unallocating the memory////////////////////////  
		 ////////////////////////////////////////////////////////////////////
       }// NULL if ended
                 free( u_self ) ;             
                 free( u_real ) ;
                 free( F_k ) ;
                 free( F_r ) ;
                 free( F_d ) ;
                 free( Rforces ) ; 
                 free( Kforces ) ;
//                 free( Dforces ) ;

      }// GPU If ended
        
                  ////////////////////////////////////////////////////////////////////  
                  ///////////////////////////Using CPU////////////////////////////////                                   
		  ////////////////////////////////////////////////////////////////////   

                 if( 0 == gpu )
  
                 #endif
                  {
                   time_t start,end;
                   if (NULL != tpot)
                    {
                     *tpot = 0 ;
                      u_s = u_d = u_r = u_k = 0 ;
                      vect[0] = vect[1] = vect[2] = 0;
                      dd = sqdd = Q = 0 ;
                  ////////////////////////////////////////////////////////////////////  
                  ////////// //Potential from the self interaction term///////////////              
		  ////////////////////////////////////////////////////////////////////   
                     
           //      int my_id ;
           //      double *dummy_array ;
          //       dummy_array = calloc ( 6  , sizeof (double));
          //       #pragma omp parallel private(my_id,i) 
         //        {
         //           my_id = omp_get_thread_num() ;
         //            for (i = my_id ; i < nbody ; i += 6)
         //              {
	 //                dummy_array[my_id] += q[i] * q[i] ;
         //              }
         //        }
         //           for (i = 0 ; i < 6 ; i++)
         //              {
         //                Q += dummy_array[i];
         //              }
	 //                u_s = -(alpha /sqrt(M_PI)) * Q ;
         //                free (dummy_array);

                  ////////////////////////////////////////////////////////////////////  
                  //////////////Potential from the Surface Dipole term ///////////////                           
		  ////////////////////////////////////////////////////////////////////   
       
        //         double  vect0 = 0 , vect1 = 0 , vect2 = 0 ;
        //         u_d = 0 ; shift_coord0 = 0 ; shift_coord1 = 0 ;  shift_coord2 = 0 ;
                 
//                  #pragma omp parallel for private(shift_coord0,shift_coord1,shift_coord2,vect0,vect1,vect2,dd,i,j)                 
          //        for (i = 0 ; i < nbody ; i++)
          //         {
          //         for (j = 0 ; j < nbody ; j++)
	  //          {
	  //          shift_coord0 = cq[j][0] - lattice[0] * min_img * floor((( cq[j][0] - cq[i][0] ) /lattice[0] ) + (1.0/2.0)) ;
	  //          shift_coord1 = cq[j][1] - lattice[1] * min_img * floor((( cq[j][1] - cq[i][1] ) /lattice[1] ) + (1.0/2.0)) ;
         //           shift_coord2 = cq[j][2] - lattice[2] * min_img * floor((( cq[j][2] - cq[i][2] ) /lattice[2] ) + (1.0/2.0));
               
	 //           vect0 = cq[i][0] * shift_coord0 ;
	 //           vect1 = cq[i][1] * shift_coord1 ;
	 //           vect2 = cq[i][2] * shift_coord2 ;
           
	 //           dd = q[i] * q[j] * ( vect0 + vect1 + vect2 ) ;
         //           u_d += ((2 * M_PI) / ( 3 * Volume ) ) * dd ;
         //           }  
         //         }
                   
                // printf("dipole-term % le\n",u_d);
                 ////////////////////////////////////////////////////////////////////  
		 /////////////////////Real-space potential CPU///////////////////////  
		 ////////////////////////////////////////////////////////////////////



                 r0 = r1 = r2 = rcut ;
                 a = 0 ;

                 shift_coord0 = shift_coord1 = shift_coord2 = 0 ;
 
        //         struct timespec tht1,tht2;
       //          clock_gettime(CLOCK_REALTIME, &tht1);
  //               #pragma omp parallel for private(shift_coord0,shift_coord1,shift_coord2,dd,sum,sqdd,a, n0,n1,n2,i,j) 
                 for (i = 0 ; i < nbody ; i++)
                  {
                   double vect0, vect1, vect2 ;
                   for (j = 0 ; j < nbody ; j++)
                    {	  
 	             shift_coord0 = cq[j][0] - lattice[0] * min_img * floor((( cq[j][0] - cq[i][0] ) /lattice[0] ) + (1.0/2.0)) ;
 	             shift_coord1 = cq[j][1] - lattice[1] * min_img * floor((( cq[j][1] - cq[i][1] ) /lattice[1] ) + (1.0/2.0)) ;
 	             shift_coord2 = cq[j][2] - lattice[2] * min_img * floor((( cq[j][2] - cq[i][2] ) /lattice[2] ) + (1.0/2.0)) ; 
	             for(n0 = -r0 ; n0 < r0 + 1.0 ; n0++)
	              {  
                      for(n1 = -r1 ; n1 < r1 + 1.0 ; n1++)
	               {
	               for(n2 = -r2 ; n2 < r2 + 1.0 ; n2++)
	                {
	                 sum = fabs(n0) + fabs(n1) + fabs(n2) ;
	                 if (( i != j) || (sum != 0))
		          {
		           vect0 = cq[i][0] - shift_coord0 + n0 * lattice[0] ;
		           vect1 = cq[i][1] - shift_coord1 + n1 * lattice[1] ;
		           vect2 = cq[i][2] - shift_coord2 + n2 * lattice[2] ;
		           dd = (vect0 * vect0 + vect1 * vect1 + vect2 * vect2) ;
		           sqdd = sqrt(dd + (rp * rp)) ;
		           a = (q[i] * q[j]/ sqdd) * erfcf( alpha * sqdd ) ;
		           potential[i] += ( 1.0 / 2.0 ) * a ;
		          }//endif
	                } /*end for j loop*/
	               }  /*end for i loop*/
	             } /*end for n2 loop*/
	           }/*end for n1 loop*/
                 }/*end for n0 loop*/
    
       //         clock_gettime(CLOCK_REALTIME, &tht2);
//              fprintf(stderr,"Real-Space Potential %g\n",diff_time(tht1,tht2));   
                for (i = 0 ; i < nbody ; i++) 
                 {
                  u_r +=  potential[i]  ;
                 }

                 ////////////////////////////////////////////////////////////////////  
		 ////////////////////////K-space potential CPU///////////////////////  
		 ////////////////////////////////////////////////////////////////////

         //        r0 = r1 = r2 = kcut ;
         //        struct timespec tht3,tht4;
           //      clock_gettime(CLOCK_REALTIME, &tht3);
             //    #pragma omp parallel for private( rvect1,rvect2,rvect3,k2,d,n0 , n1 , n2 , i)
             //        for(n0 = -r0 ; n0 < r0 + 1 ; n0++)
             //           {
             //            for(n1 = -r1 ; n1 < r1 + 1 ; n1++)
             //               {
             //                for(n2 = -r2 ; n2 < r2 + 1 ; n2++)
             //                   {
             //                    f = g = 0 ;  
      //  	                 rvect1 = ( 2.0 * M_PI * n0 ) / lattice[0] ;
//	                         rvect2 = ( 2.0 * M_PI * n1 ) / lattice[1] ;
//	                         rvect3 = ( 2.0 * M_PI * n2 ) / lattice[2] ;
//	                         k2 = rvect1 * rvect1 + rvect2 * rvect2 + rvect3 * rvect3 ;
//	                         if ( k2 != 0 )
//	                            {
//	                              d = exp(-k2 / (4.0 * alpha * alpha)) / k2;
//	                              for (i = 0 ; i < nbody ; i++)
//	                                 { 
  // 	                                  f += q[i] * cos(rvect1 * cq[i][0] + rvect2 * cq[i][1] + rvect3 * cq[i][2]) ;
   //	                                  g += q[i] * sin(rvect1 * cq[i][0] + rvect2 * cq[i][1] + rvect3 * cq[i][2]) ;
//	                                 } /*end for i loop*/
//	                              u_k += (( 2 * M_PI ) /  Volume ) * d * (( f * f ) + ( g * g )) ;
//	                            }  //endif 
 //	                         else
 //	                            {
 //	                              n0 = n1 = n2 = kcut ;
 //	                            }
//	                         } /*end for n2 loop*/
  //                          } /*end for n1 loop*/
    //                    }/*end for n0 loop*/
      //           clock_gettime(CLOCK_REALTIME, &tht4);
//               fprintf(stderr,"K-sapce Potential %g\n",diff_time(tht3,tht4));  
                 ////////////////////////////////////////////////////////////////////  
		 ///////Coordinates of all the particles in different shells/////////  
		 ////////////////////////////////////////////////////////////////////
 
/*               r0 = r1 = r2 = rcut ;
                 FILE* coordinates_out ;
                 coordinates_out = fopen( "All_Charges_coordinates.dat", "w" ) ;
                      for( n0 = -r0 ; n0 < r0 + 1 ; n0++ )
 		         {
 			   for( n1 = -r1 ; n1 < r1 + 1 ; n1++ )
 			    {
 			       for( n2 = -r2 ; n2 < r2 + 1 ; n2++ )
 			         {     
 				   for (i = 0 ; i < N; i++)
                                        {
 	                                 vect[0] = cq[i][0] + n0 * lattice[0] ;
                                         vect[1] = cq[i][1] + n1 * lattice[1] ;
                                         vect[2] = cq[i][2] + n2 * lattice[2] ;
 					fprintf( coordinates_out , "% le % le % le\n", vect[0] , vect[1] , vect[2] ) ; 
 				       }
 				 }
 			    }
 			 }

                 fclose( coordinates_out ) ;*/

                 ////////////////////////////////////////////////////////////////////  
		 ////////////////////Summing up all the Potentials///////////////////  
		 ////////////////////////////////////////////////////////////////////
		 
              //   *tpot = u_r + u_k*2 + u_s + u_d  ;
                 *tpot = u_r ;

              //   printf("r_pot % le\n",u_r);
              //   printf("k_pot % le\n",u_k*2);
        //         printf("s_pot % le\n",u_s);
          //       printf("d_pot % le\n",u_d);
                 }

                 ////////////////////////////////////////////////////////////////////  
		 ////////////////////////Real-space Forces CPU///////////////////////  
		 ////////////////////////////////////////////////////////////////////
     
                 if (NULL != cpyaq)
                  {	     
                   /*Forces due to the real space*/
	
                   r0 = r1 = r2 = rcut ;	
                   a = b = c = n = dd = sqdd = i = j = 0;
  //                 struct timespec tht5,tht6;
    //               clock_gettime(CLOCK_REALTIME, &tht5);
     //            #pragma omp parallel for private(shift_coord0,shift_coord1,shift_coord2 ,dd,sum,sqdd,e, n0,n1,n2,i,j) 
                   for( i = 0 ; i < nbody ; i++ )
                    {
                     e = f = g = h = 0;
                     double vect0, vect1, vect2 ;
                     for( j = 0 ; j < nbody ; j++ )
                      {
	               shift_coord0 = cq[j][0] - lattice[0] * min_img * floor((( cq[j][0] - cq[i][0] ) /lattice[0] ) + (1.0/2.0)) ;
 	               shift_coord1 = cq[j][1] - lattice[1] * min_img * floor((( cq[j][1] - cq[i][1] ) /lattice[1] ) + (1.0/2.0)) ;
 	               shift_coord2 = cq[j][2] - lattice[2] * min_img * floor((( cq[j][2] - cq[i][2] ) /lattice[2] ) + (1.0/2.0)) ;
	               for(n0 = -r0 ; n0 < r0 + 1 ; n0++)
	                {
                        for(n1 = -r1 ; n1 < r1 + 1 ; n1++)
	                 {
                         for(n2 = -r2 ; n2 < r2 + 1 ; n2++)
	                  {
	                   sum = fabs(n0) + fabs(n1) + fabs(n2) ;
	                   if ( (i != j) || (sum != 0) )
	                    {
	                     vect0 = cq[i][0] - shift_coord0 + n0 * lattice[0] ;
                             vect1 = cq[i][1] - shift_coord1 + n1 * lattice[1] ;
                             vect2 = cq[i][2] - shift_coord2 + n2 * lattice[2] ;	                            
		             dd = (vect0 * vect0 + vect1 * vect1 + vect2 * vect2 + (rp * rp)) ;
	                     sqdd = sqrt(dd) ;
		             e = (erfcf( alpha * sqdd ) / sqdd) + (2.0 * alpha / sqrt( M_PI )) * exp(-alpha * alpha * dd) ;
		             F_r[i][0] +=  q[i] * q[j] * e * ( vect0 / dd ) ;
		             F_r[i][1] +=  q[i] * q[j] * e * ( vect1 / dd ) ;
		             F_r[i][2] +=  q[i] * q[j] * e * ( vect2 / dd ) ;
	                    }     
	                  }/*end for n2 loop*/
	                 }/*end for n1 loop*/
	                }/*end for n0 loop*/
	              }/*end for j loop*/
                    }/*end for i loop*/		     
      //           clock_gettime(CLOCK_REALTIME, &tht6);
//               fprintf(stderr,"Real-sapce Forces %g\n",diff_time(tht5,tht6));      
   
                 ////////////////////////////////////////////////////////////////////  
		 ////////////////////////K-space Forces CPU//////////////////////////  
		 ////////////////////////////////////////////////////////////////////

        //         r0 = r1 = r2 = kcut ;
          //       a = b = c = n = dd = sqk = k2 = i = j = rpot = 0;
            //     double rpot1 , rpot2;   

             //    struct timespec tht7,tht8;
             //    clock_gettime(CLOCK_REALTIME, &tht7);
        //         #pragma omp parallel for private(k2,sqdd,a, n0,n1,n2,rvect1,rvect2,rvect3,rpot,rpot1,rpot2,i,j) 
           //      for( i = 0 ; i < nbody ; i++ )
           //       {    
           //        rpot = rpot1 = rpot2 = a = 0;
           //        for(n0 = -r0 ; n0 < r0 + 1 ; n0++)
       // 	    {
	 //            for(n1 = -r1 ; n1 < r1 + 1 ; n1++)
	   //           {
	     //          for(n2 = -r2 ; n2 < r2 + 1 ; n2++)
	       //         {	      
	        
                 //        double vect0, vect1, vect2 ;
                   //      for( j = 0 ; j < nbody ; j++ )
                   //       {
//		           shift_coord0 = cq[j][0] - lattice[0] * min_img * floor((( cq[j][0] - cq[i][0] ) /lattice[0] ) + (1.0/2.0)) ;
//		           shift_coord1 = cq[j][1] - lattice[1] * min_img * floor((( cq[j][1] - cq[i][1] ) /lattice[1] ) + (1.0/2.0)) ;
//		           shift_coord2 = cq[j][2] - lattice[2] * min_img * floor((( cq[j][2] - cq[i][2] ) /lattice[2] ) + (1.0/2.0)) ;
		     //      double cc , dd , ee ;
	             //      cc = dd = ee = 0 ;
	             //      rvect1 = ( 2.0 * M_PI * n0 ) / lattice[0] ;
	             //      rvect2 = ( 2.0 * M_PI * n1 ) / lattice[1] ;
	             //      rvect3 = ( 2.0 * M_PI * n2 ) / lattice[2] ;
	             //      k2 = rvect1 * rvect1 + rvect2 * rvect2 + rvect3 * rvect3 ; 
		       //    if (k2 != 0 )
		      //      {
		        //     vect0 = cq[i][0] - cq[j][0] ;
		       //      vect1 = cq[i][1] - cq[j][1] ;
		       //      vect2 = cq[i][2] - cq[j][2] ;
		       //      a = q[j] * exp( -k2 /( 4.0 * alpha * alpha)) * sin(rvect1 * vect0 + rvect2 * vect1 + rvect3 * vect2);
		         //    rpot += ((4 * M_PI * rvect1) / k2) * a ;
		         //    rpot1 += ((4 * M_PI * rvect2) / k2) * a ;
		         //    rpot2 += ((4 * M_PI * rvect3) / k2) * a ;
		         //   }/*end if*/	
   		      //    else
   		        //    {
  		        //    n0 = n1 = n2 = kcut ;
   		        //    }
    	             //     }/*end for n2 loop*/
    	      //          }/*end for n1 loop*/
     	      //        }/*end for n0 loop*/
        //	    }/*end for j loop*/
    	  //         F_k[i][0] = (q[i] / Volume)  * rpot*2   ; 
      	    //       F_k[i][1] = (q[i] / Volume)  * rpot1*2  ; 
     	      //     F_k[i][2] = (q[i] / Volume)  * rpot2*2  ;

           //       }/*end for i loop*/
           //      clock_gettime(CLOCK_REALTIME, &tht8);
//               fprintf(stderr,"K-sapce Forces %g\n",diff_time(tht7,tht8));

        
                 ////////////////////////////////////////////////////////////////////  
		 /////////////////Forces due to Surface Dipole term//////////////////  
		 ////////////////////////////////////////////////////////////////////
 
         //        a = b = c  = sqdd = i = j = 0;
         //        for( i = 0 ; i < nbody ; i++ )
          //        {
          //         a = b = c = 0 ;
          //       for( j = 0 ; j < nbody ; j++ )
           //         {
	    //         shift_coord0 = cq[j][0] - lattice[0] * min_img * floor((( cq[j][0] - cq[i][0] ) /lattice[0] ) + (1.0/2.0)) ;
 	      //       shift_coord1 = cq[j][1] - lattice[1] * min_img * floor((( cq[j][1] - cq[i][1] ) /lattice[1] ) + (1.0/2.0)) ;
 	     //        shift_coord2 = cq[j][2] - lattice[2] * min_img * floor((( cq[j][2] - cq[i][2] ) /lattice[2] ) + (1.0/2.0)) ; 
	      //       a += q[j] *  shift_coord0 ;
	      //       b += q[j] *  shift_coord1 ;
	      //       c += q[j] *  shift_coord2 ;
              //      }
              //     F_d[i][0] =  ((4 * M_PI * q[i]) / ( 3 * Volume ) ) * a ;
              //     F_d[i][1] =  ((4 * M_PI * q[i]) / ( 3 * Volume ) ) * b ;
             //      F_d[i][2] =  ((4 * M_PI * q[i]) / ( 3 * Volume ) ) * c ;
             //     }

                 ////////////////////////////////////////////////////////////////////  
		 //////////////////////Summing up the forces/////////////////////////  
		 ////////////////////////////////////////////////////////////////////

  	      //   for (i = 0 ; i < nbody ; i++)
              //    {
	     //      for (j = 0 ; j < 3 ; j++)
	     //       {
 		//     cpyaq[i][j] = F_r[i][j] + F_k[i][j] - F_d[i][j] ;
               //       cpyaq[i][j] = F_r[i][j] ;
	      //      }
	      //    }     
             //    }	   

  	           for (i = 0 ; i < nbody ; i++)
                     {
                      cpyaq[i][1] = F_r[i][1] ;
                      cpyaq[i][2] = F_r[i][2] ;
                      cpyaq[i][3] = F_r[i][3] ;
                     }
                  }

     
                 ////////////////////////////////////////////////////////////////////  
		 ////////////////////Unallocating the memory/////////////////////////  
		 ////////////////////////////////////////////////////////////////////
        
                 free( u_self ) ;             
                 free( u_real ) ;
                 free( F_k ) ;
                 free( F_r ) ;
                 free( F_d ) ;
                 }
  }    
/********************************************************************************************************************/		     
	     	    

    
    int bforcepbc(double *q , double (*cq)[3] , double (*cpyaq)[3] , double *potential , double *tpot , int N , double a , double b , double c , int rcut , double rp , int min_img)
    {
      
      
    int i , j , nbody , n0 , n1 , n2 , r0 , r1 , r2 , alpha ;
    double  (*F_r)[3] ;
    double  vect[3] , lattice[3] , dc , dd , sqdd ;
    double  Volume  , d , e , f , g , h , sum , u_t, doublecount,cpot,xxx,summing ;
    time_t start,end;
    double shift_coord0 , shift_coord1 , shift_coord2 ;
    double vect0, vect1, vect2 ;
    lattice[0] = a ;
    lattice[1] = b ;
    lattice[2] = c ;
    dc = 1 ;
    nbody = N ;
    u_t = 0 ;
    Volume = lattice[0] * lattice[1] * lattice[2];
    r0 = r1 = r2 = rcut ; 
    F_r = calloc( nbody * 3 , sizeof(double) ) ;
    alpha = 0.0 ;
    
                 ////////////////////////////////////////////////////////////////////  
		 ////////////////////Brute Potential using CPU///////////////////////  
		 ////////////////////////////////////////////////////////////////////
        
  if (NULL != tpot)
   
   {    
    *tpot = 0 ; 
    /*Calculating the total energy using Brute Force*/
    a = b = c  = dd = sqdd = i = j = 0;
   struct timespec tht1,tht2;
   clock_gettime(CLOCK_REALTIME, &tht1);
 //  #pragma omp parallel for private(shift_coord0,shift_coord1,shift_coord2,dd,sum,sqdd,a, n0,n1,n2,i,j)
    for (i = 0 ; i < nbody; i++)
     {
  if (i == 78 || i == 184)
  {      
     for ( j = 0 ; j < nbody ; j++ )
       {
           if (j == 78 || j == 184 || j > 293 )
           {           
	 shift_coord0 = cq[j][0] - lattice[0] * min_img * floor((( cq[j][0] - cq[i][0] ) /lattice[0] ) + 0.5) ;
 	 shift_coord1 = cq[j][1] - lattice[1] * min_img * floor((( cq[j][1] - cq[i][1] ) /lattice[1] ) + 0.5) ;
 	 shift_coord2 = cq[j][2] - lattice[2] * min_img * floor((( cq[j][2] - cq[i][2] ) /lattice[2] ) + 0.5) ;
	for( n0 = -r0 ; n0 < r0 + 1 ; n0++ )
	 {
	  for( n1 = -r1 ; n1 < r1 + 1 ; n1++ )
	   {
	    for( n2 = -r2 ; n2 < r2 + 1 ; n2++ )
	     {   
	      sum = fabs(n0) + fabs(n1) + fabs(n2);
	      if ( (i != j) || (sum != 0) )
		{
                vect0 = vect1 = vect2 = 0 ;
		vect0 = cq[i][0] - shift_coord0 + n0 * lattice[0] ;
		vect1 = cq[i][1] - shift_coord1 + n1 * lattice[1] ;
		vect2 = cq[i][2] - shift_coord2 + n2 * lattice[2] ;
		dd = (vect0 * vect0 + vect1 * vect1 + vect2 * vect2 + rp * rp) ;
		sqdd = sqrt(dd) ;
		a = (q[i] * q[j] / sqdd) * erfcf( alpha * sqdd ) ;
		potential[i] += 0.5 * a ;
		}//endif
	      }/*end for n1 loop*/
	    }/*end for n2 loop*/
	  } /*end for n0 loop*/ 
       } /*if (j == 78 || j == 184 || j > 293 )*/       
       } /*end for j loop*/                   
     } /*if (i == 78 || i == 184)*/

    if (i != 78 || i != 184)
      {
     for ( j = 0 ; j < nbody ; j++ )
       {
	 shift_coord0 = cq[j][0] - lattice[0] * min_img * floor((( cq[j][0] - cq[i][0] ) /lattice[0] ) + 0.5) ;
 	 shift_coord1 = cq[j][1] - lattice[1] * min_img * floor((( cq[j][1] - cq[i][1] ) /lattice[1] ) + 0.5) ;
 	 shift_coord2 = cq[j][2] - lattice[2] * min_img * floor((( cq[j][2] - cq[i][2] ) /lattice[2] ) + 0.5) ;
	for( n0 = -r0 ; n0 < r0 + 1 ; n0++ )
	 {
	  for( n1 = -r1 ; n1 < r1 + 1 ; n1++ )
	   {
	    for( n2 = -r2 ; n2 < r2 + 1 ; n2++ )
	     {   
	      sum = fabs(n0) + fabs(n1) + fabs(n2);
	      if ( (i != j) || (sum != 0) )
		{
                vect0 = vect1 = vect2 = 0 ;
		vect0 = cq[i][0] - shift_coord0 + n0 * lattice[0] ;
		vect1 = cq[i][1] - shift_coord1 + n1 * lattice[1] ;
		vect2 = cq[i][2] - shift_coord2 + n2 * lattice[2] ;
		dd = (vect0 * vect0 + vect1 * vect1 + vect2 * vect2 + rp * rp) ;
		sqdd = sqrt(dd) ;
		a = (q[i] * q[j] / sqdd) * erfcf( alpha * sqdd ) ;
		potential[i] += 0.5 * a ;
		}//endif
	      }/*end for n1 loop*/
	    }/*end for n2 loop*/
	  } /*end for n0 loop*/
        } /*end for j loop*/
      }  /* if (i != 78 || i != 184)*/
}/*end for i loop*/
    clock_gettime(CLOCK_REALTIME, &tht2);
//  fprintf(stderr,"CPU_Brute-Ptential took %g\n",diff_time(tht1,tht2));

    for (i = 0 ; i < nbody ; i++) { *tpot += potential[i] ;  }

    }

	     	 
	     	 ////////////////////////////////////////////////////////////////////  
		 ////////////////////////Brute Force using CPU///////////////////////  
		 ////////////////////////////////////////////////////////////////////

    if (NULL != cpyaq)
   
     {


      cpot = dd = sqdd = i = j = 0;
      struct timespec tht3,tht4;
      clock_gettime(CLOCK_REALTIME, &tht3);
     // #pragma omp parallel for private(shift_coord0,shift_coord1,shift_coord2,dd,sum,sqdd,cpot, n0,n1,n2,i,j)
      for (i = 0 ; i < nbody ; i++)
       {
        double vect0, vect1, vect2 ;
         
         if(i == 78 || i == 184)
         {
             for (j = 0 ; j < nbody ; j++)
                 {
                 if(j == 78 || j == 184 || j > 293)
                   {
                   shift_coord0 = cq[j][0] - lattice[0] * min_img * floor((( cq[j][0] - cq[i][0] ) /lattice[0] ) + 0.5) ;
                   shift_coord1 = cq[j][1] - lattice[1] * min_img * floor((( cq[j][1] - cq[i][1] ) /lattice[1] ) + 0.5) ;
                   shift_coord2 = cq[j][2] - lattice[2] * min_img * floor((( cq[j][2] - cq[i][2] ) /lattice[2] ) + 0.5) ;
                    for( n0 = -r0 ; n0 < r0 + 1 ; n0++ )
                     {
                      for( n1 = -r1 ; n1 < r1 + 1 ; n1++ )
                       {
                        for( n2 = -r2 ; n2 < r2 + 1 ; n2++ )
                         {    
                          sum = fabs(n0) + fabs(n1) + fabs(n2) ;
                          if ( (i != j) || (sum != 0) )
                           {
                            vect0 = cq[i][0] - shift_coord0 + n0 * lattice[0] ;
                            vect1 = cq[i][1] - shift_coord1 + n1 * lattice[1] ;
                            vect2 = cq[i][2] - shift_coord2 + n2 * lattice[2] ;
                            
                            dd = (vect0 * vect0 + vect1 * vect1 + vect2 * vect2+ (rp * rp)) ;
                            sqdd = sqrt(dd);
                           // cpot = (q[i] * q[j]) / dd ;
                          //  cpyaq[i][0] += cpot  * vect0 ;
                          //  cpyaq[i][1] += cpot  * vect1 ;
                          //  cpyaq[i][2] += cpot  * vect2 ;
                          cpot = (erfcf( alpha * sqdd ) / sqdd) + (2.0 * alpha / sqrt( M_PI )) * exp(-alpha * alpha * dd);
                          cpyaq[i][0] +=  q[i] * q[j] * cpot * ( vect0 / dd ) ;
                          cpyaq[i][1] +=  q[i] * q[j] * cpot * ( vect1 / dd ) ;
                          cpyaq[i][2] +=  q[i] * q[j] * cpot * ( vect2 / dd ) ;
                           } /* if ( (i != j) || (sum != 0) )*/
                         }/*end for n2 loop*/
                       } /*end for n1 loop*/
                     } /*end for n0 loop*/
                 } /*if (j == 78 || j == 184 || j > 293)*/
               } /*end for j loop*/
             } /*if(i == 78 || i == 184)*/

       if(i != 78 || i != 184)
        {
        for (j = 0 ; j < nbody ; j++)
         {
	 shift_coord0 = cq[j][0] - lattice[0] * min_img * floor((( cq[j][0] - cq[i][0] ) /lattice[0] ) + 0.5) ;
 	 shift_coord1 = cq[j][1] - lattice[1] * min_img * floor((( cq[j][1] - cq[i][1] ) /lattice[1] ) + 0.5) ;
 	 shift_coord2 = cq[j][2] - lattice[2] * min_img * floor((( cq[j][2] - cq[i][2] ) /lattice[2] ) + 0.5) ;
      	 for( n0 = -r0 ; n0 < r0 + 1 ; n0++ )
	  {
	   for( n1 = -r1 ; n1 < r1 + 1 ; n1++ )
	    {
	     for( n2 = -r2 ; n2 < r2 + 1 ; n2++ )
	      {   
	       sum = fabs(n0) + fabs(n1) + fabs(n2) ;
	       if ( (i != j) || (sum != 0) )
	        {
	         vect0 = cq[i][0] - shift_coord0 + n0 * lattice[0] ;
	         vect1 = cq[i][1] - shift_coord1 + n1 * lattice[1] ;
	         vect2 = cq[i][2] - shift_coord2 + n2 * lattice[2] ;
	     
                 dd = (vect0 * vect0 + vect1 * vect1 + vect2 * vect2+ (rp * rp)) ;
	         sqdd = sqrt(dd);
	        // cpot = (q[i] * q[j]) / dd ;
	       //  cpyaq[i][0] += cpot  * vect0 ;
	       //  cpyaq[i][1] += cpot  * vect1 ;
	       //  cpyaq[i][2] += cpot  * vect2 ;
               cpot = (erfcf( alpha * sqdd ) / sqdd) + (2.0 * alpha / sqrt( M_PI )) * exp(-alpha * alpha * dd);
               cpyaq[i][0] +=  q[i] * q[j] * cpot * ( vect0 / dd ) ;
               cpyaq[i][1] +=  q[i] * q[j] * cpot * ( vect1 / dd ) ;
               cpyaq[i][2] +=  q[i] * q[j] * cpot * ( vect2 / dd ) ; 
	        } /* if ( (i != j) || (sum != 0) )*/
	      }/*end for n2 loop*/
	    } /*end for n1 loop*/
	  } /*end for n0 loop*/  
  	 } /*end for j loop*/ 
       } /*if(i != 78 || i != 184)*/ 
      } /*end for i loop*/ 
      clock_gettime(CLOCK_REALTIME, &tht4);
//    fprintf(stderr,"CPU_Brute-forces took %g\n",diff_time(tht3,tht4));

     }    
    free( F_r ) ;
      
    }
