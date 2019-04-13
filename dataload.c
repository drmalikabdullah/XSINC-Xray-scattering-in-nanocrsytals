#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/stat.h>
#include <string.h>
#include "dataload.h"


int coordinates_atoms( int N, double (*cq)[3] , char *ee) 
  {
      FILE *in1_file ;
      int r =0;
     // char ee_test[2000];
     // strcpy(ee_test,"/scratch2/amalik/setup/proj/i3cSuc.TD.5/data/i3cSuc.TD.5.F2.280e+10.R0000001/snp/00000000/r.dat") ;
      in1_file = fopen (ee, "r");   // read only 
//    printf("%s \n",ee);
      
      if (!in1_file)
       {
          printf ("Coordinates File is Missing from %s\n", ee);
          exit (-1); 
       }

      for ( r = 0; r < N; r++ )
       {
          fscanf (in1_file, "%le %le %le", &cq[r][0], &cq[r][1], &cq[r][2]);
          cq[r][0] = cq[r][0] * 1e+10 ;
          cq[r][1] = cq[r][1] * 1e+10 ;
          cq[r][2] = cq[r][2] * 1e+10 ;
       }
          fclose (in1_file);
          return(0);
   }


int species_atoms(int N , int *atyp , char *ee) 
    {
        FILE *in2_file;
        int r = 0.0 ;
        in2_file = fopen (ee, "r");   // read only 
 //     printf("%s \n",ee);
        if (!in2_file)        // equivalent to saying if ( in_file == NULL ) 
           {
               printf ("Atom Type File is Missing from %s\n", ee);
               exit (-1);
           }
        for ( r = 0; r < N; r++ )
           {
              fscanf (in2_file, "%d", &atyp[r]);
           }
              fclose (in2_file);
              return(0);
    }


int f0_atoms(int N ,int rcount , int lcount ,double (*f0)[lcount] , char *ee)
    {
        FILE *in3_file;
        int r = 0 , rr = 0;
     //   char ee_test[2000];
     //   strcpy(ee_test,"/scratch2/amalik/setup/proj/i3cSuc.TD.5/data/i3cSuc.TD.5.F2.280e+10.R0000001/snp/00000000/f0.dat") ;
        in3_file = fopen (ee, "r");   // read only  
  //    printf("%s \n",ee);
        
        if (!in3_file)        // equivalent to saying if ( in_file == NULL )  
            {
            printf ("Form Factor File is Missing from %s\n", ee);
            exit (-1);
            }
        
        for (r = 0; r < rcount; r++)
        {
            for (rr = 0; rr < lcount; rr++)  
              {
                  fscanf (in3_file, "%le", &f0[r][rr]);
              }
        }
        fclose (in3_file);
        return(0);
    }



int Q_atoms(int N , int lcount ,double *Q , char *ee)
    {
        FILE *in4_file;
        int r = 0 ;
        in4_file = fopen (ee, "r");   // read only  
  //    printf("%s \n",ee);
        
        if (!in4_file)        // equivalent to saying if ( in_file == NULL )  
            {
              printf ("Q File is Missing from %s\n", ee);
              exit (-1);
            }
        
        for (r = 0; r < lcount; r++)
            {
              fscanf (in4_file, "%le\n", &Q[r]); 
            }
              fclose (in4_file);
              return(0);
    }

int fpp_atoms(int rcount , double *fpp ,char *ee , int fpp_check)
   {
     
     if (fpp_check == 1 )
     {
        FILE *in5_file;
        int r =0 ;
        in5_file = fopen (ee, "r");   // read only 
 //     printf("%s \n",ee);

        if (!in5_file)        // equivalent to saying if ( in_file == NULL )  
         {
          printf ("Form Factor(F'') File is Missing from %s\n", ee); 
          exit (-1); 
         }
        for ( r = 0; r < rcount; r++ ) 
         {
              fscanf (in5_file, "%le\n", &fpp[r]);
         }
          fclose (in5_file);
          return(0);
     }
     
      if (fpp_check == 0 )
        {
        
        int r = 0 ;
        for ( r = 0; r < rcount; r++ ) 
         {
            fpp[r] = 0 ;
         }
       
          return(0);
     }
     
   }


int electron_coordinates(int N , int r_ele_variable ,  double (*r_ele)[3] , char *ee) 
   {
       FILE *in6_file;
       int r = 0 ;
       in6_file = fopen (ee, "r");   // read only 
     //  printf("%s \n",ee);

       if (!in6_file)        // equivalent to saying if ( in_file == NULL )
          {
               printf ("Electrons Coordinates File is Missing from %s\n", ee); 
               exit (-1); 
          }
        for ( r = 0; r < N; r++ ) { fscanf( in6_file, "%*s %*s %*s" ) ; } 
         for ( r = 0; r < r_ele_variable; r++ )
             {
                 fscanf (in6_file, "%le %le %le", &r_ele[r][0], &r_ele[r][1], &r_ele[r][2]);
               //  r_ele[r][0] *= 1e+10 ; 
               //  r_ele[r][1] *= 1e+10 ;
               //  r_ele[r][2] *= 1e+10 ;
             }
           fclose (in6_file);
           return(0);

   }

int charge_on_electrons(int N , int r_ele_variable , int *q_ele , char *ee) 
    {
    FILE *in7_file;
    int r = 0 ;
    in7_file = fopen (ee, "r");   // read only
   //  printf("%s \n",ee);
   //   printf("DONE \n") ;
    if (!in7_file)        // equivalent to saying if ( in_file == NULL ) 
     {
         printf ("Electrons Charge File is Missing from %s\n", ee); 
         exit (-1);
     }
     for ( r = 0; r < N; r++ ) { fscanf( in7_file, "%*s" )  ; } 
      for ( r = 0; r < r_ele_variable; r++ )
         {
             fscanf (in7_file, "%d\n" ,&q_ele[r]); 
         }
         fclose (in7_file);
         return(0);
    }



int atomic_num_atoms(int N , int *Z_atoms , char *ee) 
    {
    FILE *in8_file;
    int r = 0 ;
    in8_file = fopen (ee, "r");   // read only
   //  printf("%s \n",ee);
  //   printf("DONE \n") ;
    if (!in8_file)        // equivalent to saying if ( in_file == NULL ) 
     {
         printf ("Atomic Number File is Missing from %s\n", ee); 
         exit (-1);
     }
     for ( r = 0; r < N; r++ ) 
     {
         fscanf (in8_file, "%d\n", &Z_atoms[r]);
        // if(7 == Z_atoms[r])
        //    {
          
       //      }
      //   else
      //       {
                 
      //           Z_atoms[r] = 0 ;
      //       }


        // printf("% d ",Z_atoms[r]);fflush(stdout);


     }
         fclose (in8_file);
         return(0);
    }



int f0_mod_atoms(int N ,int rcount , int lcount ,double (*f0)[lcount] , char *ee_f0mod)
    {
        FILE *in9_file;
        int r = 0 , rr = 0;
    //    double final_sum = 0.0 ;
        in9_file = fopen (ee_f0mod, "r");   // read only  
    //    printf("%s \n",ee_f0mod);
        
        if (!in9_file)        // equivalent to saying if ( in_file == NULL )  
            {
            printf ("Form Factors Modified File is Missing from %s\n", ee_f0mod);
            exit (-1);
            }
        
        for (r = 0; r < rcount; r++)
        {
            for (rr = 0; rr < lcount; rr++)  
              {
                  fscanf (in9_file, "%le", &f0[r][rr]);
//                  printf("%le ",&f0[r][rr]);
  //                final_sum = final_sum + f0[r][rr] ;
                  
              }
        }
        fclose (in9_file);
//        printf("f0_modified %le \n",final_sum);
        return(0);
    }



int species_atoms_mod(int N , int *atyp , char *ee_atypmod) 
    {
        FILE *in10_file;
        int r = 0.0 ;
      //  int hhxx = 0.0 ;
        in10_file = fopen (ee_atypmod, "r");   // read only 
  //      printf("%s \n",ee_atypmod);
        if (!in10_file)        // equivalent to saying if ( in_file == NULL ) 
           {
               printf ("Atom Type File is Missing from %s\n", ee_atypmod);
               exit (-1);
           }
        for ( r = 0; r < N; r++ )
           {
              fscanf (in10_file, "%d", &atyp[r]);   
           //   hhxx = hhxx + atyp[r];
           }
              fclose (in10_file);
          //    printf("atom type modified %d \n",hhxx);
              return(0);
    }

