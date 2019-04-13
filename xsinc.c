#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <complex.h>
#include <sys/stat.h>
#include "dataload.h"
#include "cuda_link.h"
#include "Xsinc_version_updated.h"
#include "scatintensity.h"
#include "moleculardynamicsxsinc.h"
//#include <time.h>

extern double yrandom (double, double);
long random (void);



int main (void)
{
  int    i , j , k , l      ,
         depth_cryst        ,
         nsellx,      nselly,
         NRealiz            ,
         rl,   r ,  rr      ,
         cc,  ccc ,  dd , N ,
         N_ele , SEED ,ts   ,
         total_sc           ,  
         comp_num, comp_count,
         qc = 0 ,  my_id    ,
         num_threads        ,
         r_ele_variable = 0 ,
         st_snp             ,            
         tpshapeg ,tpshapef ,
         spi_switch          ;
 
 double  scellx , 
         scelly ,
         scellz ,
         ucellx ,
         ucelly ,
         ucellz ;
  
 double (*ccq)[3]   ,
        *flu_array  ,
        *in_fluence ,
	*Hydrogen_form_factor,
	*Helium_form_factor,
	*Lithium_form_factor,
	*Beryllium_form_factor,
	*Boron_form_factor,
	*Carbon_form_factor,
	*Nitrogen_form_factor,
	*Oxygen_form_factor,
	*Fluorine_form_factor,
	*Neon_form_factor,
	*Sodium_form_factor,
	*Magnesium_form_factor,
	*Aluminum_form_factor,
	*Silicon_form_factor,
	*Phosphorus_form_factor,
	*Sulfur_form_factor,
	*Chlorine_form_factor,
	*Argon_form_factor,
	*Potassium_form_factor,
	*Calcium_form_factor,
	*Scandium_form_factor,
	*Titanium_form_factor,
	*Vanadium_form_factor,
	*Chromium_form_factor,
	*Manganese_form_factor,
	*Iron_form_factor,
	*Cobalt_form_factor,
	*Nickel_form_factor,
	*Copper_form_factor,
	*Zinc_form_factor,
	*Gallium_form_factor,
	*Germanium_form_factor,
	*Arsenic_form_factor,
	*Selenium_form_factor,
	*Bromine_form_factor,
	*Krypton_form_factor,
	*Rubidium_form_factor,
	*Strontium_form_factor,
	*Yttrium_form_factor,
	*Zirconium_form_factor,
	*Niobium_form_factor,
	*Molybdenum_form_factor,
	*Technetium_form_factor,
	*Ruthenium_form_factor,
	*Rhodium_form_factor,
	*Palladium_form_factor,
	*Silver_form_factor,
	*Cadmium_form_factor,
	*Indium_form_factor,
	*Tin_form_factor,
	*Antimony_form_factor,
	*Tellurium_form_factor,
	*Iodine_form_factor,
	*Xenon_form_factor,
	*Cesium_form_factor,
	*Barium_form_factor,
	*Lanthanum_form_factor,
	*Cerium_form_factor,
	*Praseodymium_form_factor,
	*Neodymium_form_factor,
	*Promethium_form_factor,
	*Samarium_form_factor,
	*Europium_form_factor,
	*Gadolinium_form_factor,
	*Terbium_form_factor,
	*Dysprosium_form_factor,
	*Holmium_form_factor,
	*Erbium_form_factor,
	*Thulium_form_factor,
	*Ytterbium_form_factor,
	*Lutetium_form_factor,
	*Hafnium_form_factor,
	*Tantalum_form_factor,
	*Tungsten_form_factor,
	*Rhenium_form_factor,
	*Osmium_form_factor,
	*Iridium_form_factor,
	*Platinum_form_factor,
	*Gold_form_factor,
	*Mercury_form_factor,
	*Thallium_form_factor,
	*Lead_form_factor,
	*Bismuth_form_factor,
	*Polonium_form_factor,
	*Astatine_form_factor,
	*Radon_form_factor,
	*Francium_form_factor,
	*Radium_form_factor,
	*Actinium_form_factor,
	*Thorium_form_factor,
	*Protactinium_form_factor,
	*Uranium_form_factor,
	*Neptunium_form_factor,
	*Plutonium_form_factor,
	*Americium_form_factor,
	*Curium_form_factor,
	*Berkelium_form_factor,
	*Californium_form_factor,
	*Einsteinium_form_factor,
	*Fermium_form_factor,
	*Mendelevium_form_factor,
	*Nobelium_form_factor,
	*Lawrencium_form_factor,
	*Rutherfordium_form_factor,
	*Dubnium_form_factor,
	*Seaborgium_form_factor,
	*Bohrium_form_factor,
	*Hassium_form_factor,
	*Meitnerium_form_factor;
 
 double  flu       ,
         tflu      ,
         nph       ,
         volume    ,
         dnph      ,
         peak_flue ,
         sigma     ,
         focus     ,
         a, b, c   ,
         lambda    , 
         tau       ,
         integrated_signal,
         sp_diff   ,
         FinalSnap ,
         NSTEPS    ,
         DT        ,
         T0        ;
 
 char    sflu[1000]     ,
         snp[1000]      ,
         rliz[1000]     ,
         tstep[1000]    , 
         src[1000]      ,
         ts_out[500]    ,
         output[100]    ,
         pshape[100]    ,
         rsrc[1000]     ,
         tsrc[1000]     ,
         f0src[1000]    ,
         ee[2000]       ,
         ee_f0mod[100]  ,
         ee_atypmod[100],
         inp[1000]      ,
         coord_log[100] ,
         formfactor_log[100],
         info_log[20]   ,
         Q_log[500]     ,
         rdat[10]       ,
         Zdat[10]       ,
         tdat[10]       ,
         flunce_info[20],
         beam_info[20]  ,
         f0dat[10]      ,
         fppdat[10]     ,
         qdat[10]       , 
         field[1000]    ,
         gaussian[10]   ,
         flattop[10]    ,
         hybrid[10]     ,
         system_type[20],
         fpp_status[5]  ,
         *end_dir       ,
         *OP_dir        ,
         damaged[10]    ,
         undamaged[10]  ,
         q_charge[10]   ,
         ele_r[10]      ,
         on[5]          ,
         off[5]         ,
         *r_dir         ,
         *ff_dir        ,
         effec_f[5]     ,
         input_info[20] ,
         f0_mod[10]     ,
         MD[5]          ,
         sample_type[10],
         temporal_pulse[20],
         nano_crystal[20],
         spi[5]          ,
	 sample[100]     ;

double (*CELL)[3]; int ijk_count;
 
double complex *f_0           ,
                *qrterm_comp   ,
                *dummy_array_1 ,
                *dummy_array_2 ,
                *dummy_array_3 ,
                *dummy_array_4 ,
                *dummy_array_5 ,
                *dummy_array_6 ,
                *dummy_array_7 ;
                      
 FILE    *qpoints_out , 
         *in_file     ,
         *info_out    ,
         *dir_info    ,
         *beam_profle ,
         *fluenceout  , 
         *inputout    ;
         
         strcpy (rdat, "r.dat")           ;
         strcpy (f0_mod, "f0_mod.dat")    ;
         strcpy (Zdat, "Z.dat")           ;
         strcpy (tdat, "T.dat")           ;
         strcpy (f0dat, "f0.dat")         ;
         strcpy (qdat, "Q.dat")           ;
         strcpy (fppdat, "fpp.dat")       ; 
         strcpy (gaussian, "gaussian")       ;
         strcpy (q_charge, "q.dat")       ;
         strcpy (ele_r, "r.dat")          ;
         strcpy (flattop, "flattop")      ;
         strcpy (hybrid, "hybrid")        ;  
         strcpy (damaged, "damaged")      ;
         strcpy (undamaged, "undamaged")  ;
         strcpy (on, "on")  ;
         strcpy (off, "off")  ; 
         strcpy (nano_crystal, "nanocrystal")  ;
         strcpy (spi, "spi")  ;

	 int  Hydrogen_number_count = 0 ;
		int  Helium_number_count = 0 ;
		int  Lithium_number_count = 0 ;
		int  Beryllium_number_count = 0 ;
		int  Boron_number_count = 0 ;
		int  Carbon_number_count = 0 ;
		int  Nitrogen_number_count = 0 ;
		int  Oxygen_number_count = 0 ;
		int  Fluorine_number_count = 0 ;
		int  Neon_number_count = 0 ;
		int  Sodium_number_count = 0 ;
		int  Magnesium_number_count = 0 ;
		int  Aluminum_number_count = 0 ;
		int  Silicon_number_count = 0 ;
		int  Phosphorus_number_count = 0 ;
		int  Sulfur_number_count = 0 ;
		int  Chlorine_number_count = 0 ;
		int  Argon_number_count = 0 ;
		int  Potassium_number_count = 0 ;
		int  Calcium_number_count = 0 ;
		int  Scandium_number_count = 0 ;
		int  Titanium_number_count = 0 ;
		int  Vanadium_number_count = 0 ;
		int  Chromium_number_count = 0 ;
		int  Manganese_number_count = 0 ;
		int  Iron_number_count = 0 ;
		int  Cobalt_number_count = 0 ;
		int  Nickel_number_count = 0 ;
		int  Copper_number_count = 0 ;
		int  Zinc_number_count = 0 ;
		int  Gallium_number_count = 0 ;
		int  Germanium_number_count = 0 ;
		int  Arsenic_number_count = 0 ;
		int  Selenium_number_count = 0 ;
		int  Bromine_number_count = 0 ;
		int  Krypton_number_count = 0 ;
		int  Rubidium_number_count = 0 ;
		int  Strontium_number_count = 0 ;
		int  Yttrium_number_count = 0 ;
		int  Zirconium_number_count = 0 ;
		int  Niobium_number_count = 0 ;
		int  Molybdenum_number_count = 0 ;
		int  Technetium_number_count = 0 ;
		int  Ruthenium_number_count = 0 ;
		int  Rhodium_number_count = 0 ;
		int  Palladium_number_count = 0 ;
		int  Silver_number_count = 0 ;
		int  Cadmium_number_count = 0 ;
		int  Indium_number_count = 0 ;
		int  Tin_number_count = 0 ;
		int  Antimony_number_count = 0 ;
		int  Tellurium_number_count = 0 ;
		int  Iodine_number_count = 0 ;
		int  Xenon_number_count = 0 ;
		int  Cesium_number_count = 0 ;
		int  Barium_number_count = 0 ;
		int  Lanthanum_number_count = 0 ;
		int  Cerium_number_count = 0 ;
		int  Praseodymium_number_count = 0 ;
		int  Neodymium_number_count = 0 ;
		int  Promethium_number_count = 0 ;
		int  Samarium_number_count = 0 ;
		int  Europium_number_count = 0 ;
		int  Gadolinium_number_count = 0 ;
		int  Terbium_number_count = 0 ;
		int  Dysprosium_number_count = 0 ;
		int  Holmium_number_count = 0 ;
		int  Erbium_number_count = 0 ;
		int  Thulium_number_count = 0 ;
		int  Ytterbium_number_count = 0 ;
		int  Lutetium_number_count = 0 ;
		int  Hafnium_number_count = 0 ;
		int  Tantalum_number_count = 0 ;
		int  Tungsten_number_count = 0 ;
		int  Rhenium_number_count = 0 ;
		int  Osmium_number_count = 0 ;
		int  Iridium_number_count = 0 ;
		int  Platinum_number_count = 0 ;
		int  Gold_number_count = 0 ;
		int  Mercury_number_count = 0 ;
		int  Thallium_number_count = 0 ;
		int  Lead_number_count = 0 ;
		int  Bismuth_number_count = 0 ;
		int  Polonium_number_count = 0 ;
		int  Astatine_number_count = 0 ;
		int  Radon_number_count = 0 ;
		int  Francium_number_count = 0 ;
		int  Radium_number_count = 0 ;
		int  Actinium_number_count = 0 ;
		int  Thorium_number_count = 0 ;
		int  Protactinium_number_count = 0 ;
		int  Uranium_number_count = 0 ;
		int  Neptunium_number_count = 0 ;
		int  Plutonium_number_count = 0 ;
		int  Americium_number_count = 0 ;
		int  Curium_number_count = 0 ;
		int  Berkelium_number_count = 0 ;
		int  Californium_number_count = 0 ;
		int  Einsteinium_number_count = 0 ;
		int  Fermium_number_count = 0 ;
		int  Mendelevium_number_count = 0 ;
		int  Nobelium_number_count = 0 ;
		int  Lawrencium_number_count = 0 ;
		int  Rutherfordium_number_count = 0 ;
		int  Dubnium_number_count = 0 ;
		int  Seaborgium_number_count = 0 ;
		int  Bohrium_number_count = 0 ;
		int  Hassium_number_count = 0 ;
		int  Meitnerium_number_count = 0 ;
		int  rcut, kcut;
		char hard_wall[10] , min_img[10];
		double regparam , alphaforewald ;
 
         ////////////////////////////////////////////////////////////////////  
         /////////////////Reading the Data from Input FILE///////////////////  
         ////////////////////////////////////////////////////////////////////
    
   
       in_file = fopen("input.txt", "r"); // read only 
        while ( fgets( inp , sizeof( inp )-1, in_file) != NULL )
          {
           sscanf(inp , "%s" , field) ;
    
           if ( strcmp( field , "Data_Path") == 0 ) { sscanf(inp , "%*s %s" , src) ;  }
           
           else if ( strcmp( field , "Sample_type") == 0 ) { sscanf(inp , "%*s %s" , sample_type) ; }
           else if ( strcmp( field , "CPUcores") == 0 ) { sscanf(inp , "%*s %d" , &num_threads) ; }
           else if ( strcmp( field , "NumAtoms") == 0 ) { sscanf(inp , "%*s %d" , &N) ; }
           else if ( strcmp( field , "TotalParticles") == 0 ){ sscanf(inp , "%*s %d" , &N_ele) ; }
           else if ( strcmp( field , "NumPhotons") == 0 ) { sscanf(inp , "%*s %le" , &nph) ; }
           else if ( strcmp( field , "SpatialPulse") == 0 ) { sscanf(inp , "%*s %s" , &pshape) ; }
           else if ( strcmp( field , "TemporalPulse") == 0 ) { sscanf(inp , "%*s %s" , &temporal_pulse) ; }
           else if ( strcmp( field , "system") == 0 ) { sscanf(inp , "%*s %s" , &system_type) ; }
           else if ( strcmp( field , "FDoublePrime") == 0 ){ sscanf(inp , "%*s %s" , &fpp_status) ; }
           else if ( strcmp( field , "PulseDuration") == 0 ){sscanf(inp , "%*s %le" , &tau) ; }
           else if ( strcmp( field , "T0") == 0 ) { sscanf(inp , "%*s %le" , &T0) ; }
           else if ( strcmp( field , "EffectiveF") == 0 ) { sscanf(inp , "%*s %s" , &effec_f) ; }
           else if ( strcmp( field , "Lambda") == 0 ) { sscanf(inp , "%*s %le" , &lambda) ; }
           else if ( strcmp( field , "UdimX") == 0 ) { sscanf(inp , "%*s %le" , &ucellx) ;}
           else if ( strcmp( field , "UdimY") == 0 ) { sscanf(inp , "%*s %le" , &ucelly) ; }
           else if ( strcmp( field , "UdimZ") == 0 ) { sscanf(inp , "%*s %le" , &ucellz) ; }
           else if ( strcmp( field , "SdimX") == 0 ) { sscanf(inp , "%*s %le" , &scellx) ; }
           else if ( strcmp( field , "SdimY") == 0 ) { sscanf(inp , "%*s %le" , &scelly) ; }
           else if ( strcmp( field , "SdimZ") == 0 ) { sscanf(inp , "%*s %le" , &scellz) ; }
           else if ( strcmp( field , "Focus") == 0 ) { sscanf(inp , "%*s %le" , &focus) ; }
           else if ( strcmp( field , "Depth") == 0 ) { sscanf(inp , "%*s %d" , &depth_cryst) ; }
           else if ( strcmp( field , "StartSnap") == 0 ) { sscanf(inp , "%*s %d" , &st_snp) ; }
           else if ( strcmp( field , "SnapDiff") == 0 ) { sscanf(inp , "%*s %le" , &sp_diff) ; }
           else if ( strcmp( field , "FinalSnapshot") == 0 ) { sscanf(inp , "%*s %le" , &FinalSnap) ; }
           else if ( strcmp( field , "Realizations") == 0 ) { sscanf(inp , "%*s %d" , &NRealiz) ; }
           else if ( strcmp( field , "Molecular_Dynamics") == 0 ) { sscanf(inp , "%*s %s" , &MD) ; }
           else if ( strcmp( field , "NSTEPS") == 0 ) { sscanf(inp , "%*s %le" , &NSTEPS) ; }
           else if ( strcmp( field , "DT") == 0 ) { sscanf(inp , "%*s %le" , &DT) ; }
           else if ( strcmp( field , "regparam") == 0 ) { sscanf(inp , "%*s %le" , &regparam) ; }
           else if ( strcmp( field , "alpha") == 0 ) { sscanf(inp , "%*s %le" , &alphaforewald) ; }
           else if ( strcmp( field , "Rcut") == 0 ) { sscanf(inp , "%*s %d" , &rcut) ; }
           else if ( strcmp( field , "Kcut") == 0 ) { sscanf(inp , "%*s %d" , &kcut) ; }
           else if ( strcmp( field , "MinImg") == 0 ) { sscanf(inp , "%*s %s" , min_img) ; }
           else if ( strcmp( field , "HARDWALL") == 0 ) { sscanf(inp , "%*s %s" , hard_wall) ; }          
           else if ( strcmp( field , "Seed") == 0 ) { sscanf(inp , "%*s %d" , &SEED) ; }
           else {printf ("Incorrect Input Parameters %s\n", inp);exit(-1);}
          }
    
                fclose(in_file) ; 


     //////////////////////////////////////////////////////////////////////
     /////////////////////////////////////////////////////////////////////
     /////////////////XATOM CALL SETTING/////////////////////////////////

      //FILE *fp;
      //char path[100000];
      //fp = popen("xatom  -xmdyn -Z 53 -conf 1s2_2s2_2p6_3s2_3p5_3d6_4s1 -form_factor", "r");
      
     // if (fp == NULL) 
     // {
     //   printf("Failed to run command\n" );
     //   exit(1);
    //  }

        /* Read the output a line at a time - output it. */
      
    //  while (fgets(path, sizeof(path)-1, fp) != NULL) 
    //  {
        //printf("%s", path);
	
    //  }

       /* close */
    //  pclose(fp);

  //    return 0;
	
     //////////////////////////////////////////////////////////////////////
     /////////////////////////////////////////////////////////////////////





		////////////////////////////////////////////////////////////////////////
		//////////////////////////Molecular Dynamics Secttion///////////////////
		////////////////////////////////////////////////////////////////////////

if (strcmp ( on,MD ) == 0)
       {
	 
	 
	        //////////////////////////////////////////////////////////////////
                ////////////////////Print Outs of Inputs//////////////////////////
                //////////////////////////////////////////////////////////////////
                printf ("***********************************************************\n");
                printf ("When using XSINC, please cite\n");
		printf ("[1]  M. M. Abdullah, Z. Jurek, S.-K. Son and R. Santra\n");
		printf ("     Structural Dynamics 3, 054101 (2016)\n");                		
		printf ("[2]  M. M. Abdullah, S.-K. Son, Z. Jurek and R. Santra\n");
		printf ("     IUCrJ 5(6), 00 (2018) [10.1107/S2052252518011442]\n\n");		
	
		printf ("******************************************\n");
                printf ("COPYRIGHT BY\n");
		printf ("Deutsches Elektronen-Synchrotron DESY\n");
		printf ("Notkestr.  85\n");
		printf ("22603 Hamburg\n");
		printf ("Germany\n");
                printf ("****************************************\n\n");


		printf ("***********************************************************\n");
                printf ("INFORMATION ABOUT THE XSINC MOLECULAR DYNAMICS PARAMETERS\n");
                printf ("***********************************************************\n\n");
                printf ("\n");
		printf ("Molecular Dynamics                 =  % s\n", MD);
		printf ("Xsinc Revision on SVN              =  % s\n", SVN_VERSION);
                printf ("No. OF THREADS                     = % d\n", num_threads); 
                printf ("No. of Particles in a Super Cell   = % d\n", N);
		printf ("Number of total steps              = % f\n", NSTEPS);
		printf ("Delta time                         = % le\n", DT);
                printf ("SUPERCELL SIZE (Angstrom)          = % 1f % 1f % 1f\n", scellx, scelly, scellz);
		printf ("regparam                           = % le\n", regparam);
		printf ("alpha                              = % le\n", alphaforewald);
		printf ("Rcut                               = % d\n", rcut);
		printf ("Kcut                               = % d\n", kcut);
		printf ("MinImg                             =  % s\n", min_img);
		printf ("HARDWALL                           =  % s\n", hard_wall);

		int ii ;
		double (*v)[3] ;
		double *q  , *cpyq1 , *cpyq2 , (*BruteF)[3] , *potential;
		double (*cq)[3] , (*cq2)[3] , (*cpycq1)[3] , (*cpycq2)[3] , (*cpyaq)[3] ;
		double  tpot , sumoq ;
		FILE* out_file ;
		FILE* out_file2 ;
		q = calloc( N , sizeof(double) ) ;          /* Exporting Charges Array*/
		cq = calloc( N * 3 , sizeof(double) ) ;     /* Exporting Coordinates Array*/
		cpyaq = calloc( N * 3 , sizeof(double) ) ;  /* Importing Acceleration Array*/  
		potential = calloc( N , sizeof(double) ) ;          /* Importing Potential*/
		//md_integrator_verlet( N , cq , v , q , 0.1e-18 , 100000 , scellx , scelly , scellz , regparam , alphaforewald , rcut , kcut , select , min_img ,hard_wall) ;

		
		 ////////////////////////////////////////////////////////////////////  
		 ///////////Randomly Generated Coordinates and charges///////////////  
		 ////////////////////////////////////////////////////////////////////
		
		for (i = 0 ; i < N ; i++)
		    {
		       q[i] = yrandom(-0.1 , 0.1 ) ;
		         for ( ii = 0 ; ii < 3 ; ii++ )
			   {
                            cq[i][ii] = yrandom( -scellx/2 * 0.9 , scellx/2 * 0.9 ) ;
			   }

		    }
		    
		out_file = fopen( "Charges_coordinates.dat", "w" ) ;

		
		 ////////////////////////////////////////////////////////////////////  
		 ///////////////Writting Coordinates to Output File//////////////////  
		 ////////////////////////////////////////////////////////////////////

		for (i = 0 ; i < N ; i++)
		    {
		       fprintf( out_file , "% le % le % le\n", cq[i][0] , cq[i][1] , cq[i][2] ) ;
		    }  
  
                fclose( out_file ) ;

  
  
		for (i = 0 ; i < N/2 ; i++)
		    {
		       q[i] = -1  ;
		    }
  
		for (i = N/2 ; i < N ; i++)
		    {
		       q[i] = 1  ;    
		    }
		for(i = 0 ; i < N ; i++)
		    {
		       sumoq += q[i] ;
		    }
		       printf ("Initial total Charge               =  %le\n", sumoq);
		       
		       
		       
		 ////////////////////////////////////////////////////////////////////  
		 ///////////////////////Generating Outputs///////////////////////////  
		 ////////////////////////////////////////////////////////////////////     



		out_file = fopen( "Ewald_Forces.dat", "w" ) ;
		for (i = 0 ; i < N ; i++)
		    {
		      fprintf( out_file , "%le %le %le\n", cpyaq[i][0] , cpyaq[i][1] , cpyaq[i][2] ) ;
		    }
		fclose( out_file ) ;
		printf ("Initial total Potential            =  %le\n", tpot) ;
		       
		       md_integrator_verlet( N , cq , v , q , DT , NSTEPS , scellx , scelly , scellz , regparam , alphaforewald , rcut , kcut , 1 , 1 ,hard_wall) ;
		       
		         free(q) ;          /* Exporting Charges Array*/
			 free(cq) ;     /* Exporting Coordinates Array*/
			 free(cpyaq) ;  /* Importing Acceleration Array*/  
		         free(potential) ;          /* Importing Potential*/
       }





else
   {
		///////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////

                omp_set_num_threads(num_threads) ;
                dummy_array_1 = calloc ( num_threads  , sizeof (double complex));
                dummy_array_2 = calloc ( num_threads , sizeof (double complex));
                dummy_array_3 = calloc ( num_threads  , sizeof (double complex));
                dummy_array_4 = calloc ( num_threads  , sizeof (double complex));
                dummy_array_5 = calloc ( num_threads  , sizeof (double complex));
                dummy_array_6 = calloc ( num_threads  , sizeof (double complex));
                dummy_array_7 = calloc ( num_threads  , sizeof (double complex));
                tau = tau * 1000 ;
                T0 = T0 * 1000 ; 
                st_snp = st_snp * 1000 ;
                FinalSnap = FinalSnap * 1000 ;
                r_ele_variable = N_ele - N ;

                if (r_ele_variable == -N)
                {
                    r_ele_variable = 0.0 ;
                }


              tpshapef = 0 ;
              tpshapeg = 0 ;
                if ( strcmp( gaussian,temporal_pulse ) == 0.0 )
                   {
                    tpshapeg = 1.0 ;
                   }

                if ( strcmp( flattop,temporal_pulse ) == 0.0 )
                   {
                    tpshapef = 1.0 ;
                   }


                //////////////////////////////////////////////////////////////////
                ////////////////// Input fluence from the File////////////////////
                //////////////////////////////////////////////////////////////////
  
 
                  FILE *influ_file = fopen("input_fluence.txt","r") ;
                  int fluch , flu_count = 0 ;
                   do
                        {
                         fluch = fgetc(influ_file); if (fluch == '\n')
                         flu_count++ ;
                        }
   
                        while (fluch != EOF) ;
                        fclose(influ_file);
                  in_fluence = calloc(flu_count, sizeof (double)) ;
                  influ_file = fopen("input_fluence.txt","r"); 
      
                  if (!influ_file)	// equivalent to saying if ( in_file == NULL ) 
                        {
                         printf ("Input fluence File is Missing from %s\n");
                         exit(-1);
                        }

                  for(r = 0 ; r < flu_count ; r++)
                        {
                         fscanf ( influ_file , "%le" , &in_fluence[r] );
                        }
                         fclose(influ_file); 
   
                /////////////////////////////////////////////////////////////////
                ///////////////////////INPUT Q-POINTS////////////////////////////
                /////////////////////////////////////////////////////////////////
   
                  FILE *qpt_file = fopen("input_qpoints.txt","r");
                  int qpch , qp_rcount = 0 , qp_lcount = 0 ;
   
                   do
                        {
                         qpch = fgetc(qpt_file); if (qpch == '\n')
                         qp_rcount ++ ;
                        }
     
                  while ( qpch != EOF ) ;
                  fclose(qpt_file);
   
                  qpt_file = fopen( "input_qpoints.txt","r" );
                  while(0 == fscanf ( qpt_file, "%*s" ))
                       {
                       qp_lcount++ ;
                       }
                       fclose( qpt_file );
   
                  qp_lcount = qp_lcount / qp_rcount ;
  
                  double ( *dummy_qpt_array )[qp_lcount];
                  dummy_qpt_array = calloc( qp_rcount * qp_lcount, sizeof( double ));
   
                  qpt_file = fopen( "input_qpoints.txt","r" );
   
                   if( !qpt_file )
                       {
                        printf ( "Input Qpoint File is Missing from %s\n" );
                        exit( -1 );
                       }
   
                   for( r = 0 ; r < qp_rcount ; r++ )
                      {
                      for( rr = 0 ; rr < qp_lcount ; rr++ )
                        {
                        fscanf(qpt_file , "%le" , &dummy_qpt_array[r][rr]);
                        }
                      }
                   fclose(qpt_file);
  
                //////////////////////////////////////////////////////////////////
                ////////////////////Check for the Q-points////////////////////////
                //////////////////////////////////////////////////////////////////
  
                int rri , rrj , rrk , cx;
                double qvect11 , qvect22 , qvect33 , e_sphere , modd_rvect ;
                e_sphere = (4 * M_PI) / lambda ;
                cx = 0 ;

                for ( r = 0 ; r < qp_rcount ; r++ )
                        {
                        rri = dummy_qpt_array[r][0] ;
                        rrj = dummy_qpt_array[r][1] ;
                        rrk = dummy_qpt_array[r][2] ;

                        qvect11 = (2.0 * M_PI * rri) / ucellx;
                        qvect22 = (2.0 * M_PI * rrj) / ucelly; 
                        qvect33 = (2.0 * M_PI * rrk) / ucellz; 
        
                         modd_rvect = sqrt (qvect11 * qvect11 +
                                            qvect22 * qvect22 +
                                            qvect33 * qvect33 );
        
                        if (modd_rvect <= e_sphere)
                                {
                                  cx++ ;
                                }
                        }

                double (*qpt_array)[qp_lcount];
                qpt_array = calloc(cx * qp_lcount, sizeof(double));

                 cx = 0 ; 
                for ( r = 0 ; r < qp_rcount ; r++ )
                        {
                         rri = dummy_qpt_array[r][0] ;
                         rrj = dummy_qpt_array[r][1] ;
                         rrk = dummy_qpt_array[r][2] ;

                         qvect11 = (2.0 * M_PI * rri) / ucellx;
                         qvect22 = (2.0 * M_PI * rrj) / ucelly; 
                         qvect33 = (2.0 * M_PI * rrk) / ucellz; 
     
                         modd_rvect = sqrt (qvect11 * qvect11 +
                                            qvect22 * qvect22 +
                                            qvect33 * qvect33 );
     
                        if ( modd_rvect <= e_sphere )
                                {
                                 qpt_array[cx][0] = dummy_qpt_array[r][0] ;
                                 qpt_array[cx][1] = dummy_qpt_array[r][1] ;
                                 qpt_array[cx][2] = dummy_qpt_array[r][2] ;
                                 cx++ ;
                                }
                        }
                   qp_rcount = cx ;
                   free(dummy_qpt_array) ;

                //////////////////////////////////////////////////////////////////
                ////////////////////Grid Size Definition//////////////////////////
                //////////////////////////////////////////////////////////////////
 

                if ( strcmp( nano_crystal,sample_type ) == 0.0 )
                {
 
                volume = ucellx * ucelly * ucellz ;
                sigma = focus ;
        
                if (strcmp(gaussian,pshape) == 0.0 )
                   {
                     sigma = sigma / 2.35482004503 ; //sigma /sqrt(8 ln2)
                     nsellx = ceil (6 * sigma / scellx);
                     if ((nsellx % 2) == 0)
                        {
                        nsellx = nsellx + 1;
                        }
                      nsellx = floor (nsellx / 2);
                   }

                 if (strcmp(gaussian,pshape) == 0 )
                   {
                     nselly = ceil (6 * sigma / scelly);
                     if ((nselly % 2) == 0)
                        {
                        nselly = nselly + 1;
                        }
                      nselly = floor (nselly / 2);
                   }

                if (strcmp(flattop,pshape) == 0 )
                   {
                      nsellx = ceil (focus / scellx);
                      if ((nsellx % 2) == 0)
                         {
                             nsellx = nsellx + 1;
                         }
                      nsellx = floor (nsellx / 2.0);
                   }
   
                 if (strcmp(flattop,pshape) == 0.0 )
                   {
                      nselly = ceil (focus / scelly);
                      if ((nselly % 2) == 0.0)
                         {
                             nselly = nselly + 1.0;
                         }
                      nselly = floor (nselly / 2.0);
                   }
                }

                if ( strcmp( spi,sample_type ) == 0.0 )
                {
                    volume = ucellx * ucelly * ucellz ;
                    sigma = focus ;
                       nsellx = nsellx = nsellx = 0 ;
                    }
                 a = b = c = 0;
                //////////////////////////////////////////////////////////////////
                ////////////////////Print Outs of Inputs//////////////////////////
                //////////////////////////////////////////////////////////////////                
                printf ("***********************************************************\n");
                printf ("When using XSINC, please cite\n");
		printf ("[1]  M. M. Abdullah, Z. Jurek, S.-K. Son and R. Santra\n");
		printf ("     Structural Dynamics 3, 054101 (2016)\n");                		
		printf ("[2]  M. M. Abdullah, S.-K. Son, Z. Jurek and R. Santra\n");
		printf ("     IUCrJ 5(6), 00 (2018) [10.1107/S2052252518011442]\n\n");		
	
		printf ("******************************************\n");
                printf ("COPYRIGHT BY\n");
		printf ("Deutsches Elektronen-Synchrotron DESY\n");
		printf ("Notkestr.  85\n");
		printf ("22603 Hamburg\n");
		printf ("Germany\n");
                printf ("******************************************\n\n");
                
		printf ("***********************************************************\n");
                printf ("INFORMATION ABOUT THE XSINC SCATTERING INTENSITY PARAMETERS\n");
                printf ("***********************************************************\n\n"); 
                printf ("\n");
		printf ("Xsinc Revision on SVN              =  % s\n", SVN_VERSION);
                printf ("Type Of the Sample                 =  % s\n", sample_type); 
                printf ("Type Of the System                 =  % s\n", system_type); 
                printf ("F Double Prime                     =  % s\n", fpp_status ); 
                printf ("No. OF THREADS                     = % d\n", num_threads); 
                printf ("INPUT DATA PATH                    =  % s\n", src);
                printf ("PULSE SHAPE                        =  % s\n", pshape);
                printf ("TEMPORAL PULSE SHAPE               =  % s\n", temporal_pulse);
                printf ("SIGMA (Angstroms)                  = % 1f\n", sigma);
                printf ("EffectiveF                         =  % s\n", effec_f);
                printf ("No. of Particles per Super Cell    = % d\n", N);
                printf ("No. of the Shells of a Crystal     = % d %d\n", nsellx, nselly);
                printf ("Layers of the Depth of the Crystal = % d\n", depth_cryst);
                printf ("SUPERCELL SIZE (Angstrom)          = % 1f % 1f % 1f\n", scellx, scelly, scellz);
                printf ("QPOINTS                            = % d\n", qp_rcount );
                printf ("No OF SINGLE FLUENCE REALIZATIONS  = % d\n", NRealiz );
                printf ("TOTAL TIMESTEPS                    = % le\n", FinalSnap/sp_diff);
                printf ("PULSE DURATION                     = % .0f fs\n", tau/1000);
                printf ("CENTRE OF PULSE                    = % .0f fs\n", T0/1000);
		printf ("\n");
 		printf ("***********************************************************\n");
                 printf ("INFORMATION ABOUT THE XSINC MOLECULAR DYNAMICS PARAMETERS\n");
                 printf ("***********************************************************\n\n");
                 printf ("\n");
 		printf ("Molecular Dynamics                 =  % s\n", MD);
 		printf ("regparam                           = % le\n", regparam);
 		printf ("alpha                              = % le\n", alphaforewald);
 		printf ("Rcut                               = % d\n", rcut);
 		printf ("Kcut                               = % d\n", kcut);
 		printf ("MinImg                             =  % s\n", min_img);
 		printf ("HARDWALL                           =  % s\n", hard_wall);

                //////////////////////////Directory Setup///////////////////////////
                ////////////////////////////////////////////////////////////////////
  
                double ARRAY_COUNT = ((nsellx*2)+1)*((nselly*2)+1)*((depth_cryst)); //here 3 because k vector is from -1 t0 1
               // cq = calloc (N * 3, sizeof (double));	/* Exporting Coordinates Array */
                ccq = calloc (N * 3, sizeof (double));	/* Exporting Coordinates Array */
                
		if ( strcmp( nano_crystal,sample_type ) == 0.0 )
                   {
		     spi_switch = 1 ; 
		     flu_array = calloc(ARRAY_COUNT, sizeof (double));
	           }
	         
	        if ( strcmp( spi,sample_type ) == 0.0 ) 
                   {
		      spi_switch = 0 ;
		      flu_array = calloc(NRealiz, sizeof (double));
	           }
                mkdir("output", 0777) ;
 
                mkdir("./output/timesteps", 0777);
                mkdir("./output/positions", 0777);
                mkdir("./output/formfactor",0777);
                end_dir = src + strlen (src) ; 
                strcpy (snp, "/snp/");
                strcpy(info_log,"./output/info") ; 
 
                info_out = fopen (info_log, "w");
                strcpy(output,"./output/timesteps/") ;  
 
                OP_dir = output + strlen (output);
                strcpy(coord_log,"./output/positions/R") ;
 
                r_dir = coord_log + strlen (coord_log);
 
                strcpy(formfactor_log,"./output/formfactor/") ;
                ff_dir = formfactor_log + strlen (formfactor_log);


                
                
                ////////////////////////////////////////////////////////////////////  
                ////////////////Routine To calculate Beam Profile///////////////////  
                ////////////////////////////////////////////////////////////////////
                
                strcpy(beam_info,"./output/beam_profile") ;
                beam_profle = fopen (beam_info,"w") ;
                integrated_signal = 0 ; 
                double timestep_var = 0 ;
                double dummy_T0 = T0 / 1000 ;
                double dummy_st_snp = (st_snp/1000) - (st_snp/1000);
                double dummy_FinalSnap = (FinalSnap / 1000) - (st_snp/1000) ;
                double dummy_sp_diff = sp_diff / 1000 ;
                double dummy_tau = tau / 1000 ;


//                printf("% le %le %le %le %le %d\n", FinalSnap, sp_diff , T0 , tau , integrated_signal , tpshapef );
              
                if (1 == tpshapeg )
                 {
                    for ( timestep_var = dummy_st_snp; timestep_var <= dummy_FinalSnap; timestep_var = timestep_var + dummy_sp_diff )
                        {
                            integrated_signal += exp(-((timestep_var-dummy_T0) * (timestep_var-dummy_T0) )/((dummy_tau*dummy_tau)/(4 * log(2))));
                        }
           
                    for ( timestep_var =  dummy_st_snp; timestep_var <=  dummy_FinalSnap; timestep_var = timestep_var + dummy_sp_diff )
                        {
                            fprintf(beam_profle, "% le \n", exp(-((timestep_var-dummy_T0) * (timestep_var-dummy_T0) )/((dummy_tau*dummy_tau)/(4 * log(2))))/integrated_signal);
                        }
                        fclose(beam_profle) ;
                  }
                
                
                if (0 == tpshapeg ) 
                 {
                    integrated_signal = 1 ;
                    
                    for ( timestep_var = st_snp; timestep_var <= FinalSnap; timestep_var = timestep_var + sp_diff )
                        {
                         fprintf(beam_profle, "% d \n", tpshapef);
                        }
                        fclose(beam_profle) ;
                    }
        


                ////////////////////////////////////////////////////////////////////  
                ///////////Routine To calculate Fluence Distribution////////////////  
                ////////////////////////////////////////////////////////////////////

                srand(SEED);
                total_sc = 0 ;
                ijk_count = 0 ;
          
          if ( strcmp( nano_crystal,sample_type ) == 0.0 )
              {               
                double ff1 , ff2 ; 
                int ff3 ;
                strcpy(flunce_info,"./output/flunce_info") ;
                fluenceout = fopen (flunce_info,"w");
   
                if ( strcmp( gaussian,pshape ) == 0 )
                   {
                    peak_flue = (nph )  / (2 * M_PI * sigma * sigma) ;
                   }
   
                if ( strcmp( flattop,pshape ) == 0 )
                   {
                    peak_flue = (nph * 0.8825 )  / (sigma * sigma) ; 
                   }
   
                if ( strcmp( hybrid,pshape ) == 0 )
                   {
                    peak_flue = nph  / (sigma * sigma) ; 
                   }

                printf("Peak Fluence (1/micron^2)          = % le\n", peak_flue/1e-8 );
// 	        printf ("\n");
// 	        printf ("\n");
                total_sc = 0 ;
    
                 for ( i = -nsellx; i < nsellx + 1; i++ )
                    {
                      for ( j = -nselly; j < nselly + 1; j++ )
	                 {
                           for ( k = 0; k < depth_cryst; k++ )
	                      {
                                if ( strcmp ( gaussian , pshape ) == 0 )
                                  {  
                                   flu_array[total_sc] = nph / ( sigma*sigma *2 * M_PI )*
                                                           exp (-((i * i * scellx * scellx)+
                                                                  (j * j * scelly * scelly)
                                                                 )/(2* (sigma * sigma)));      
                                   dnph = ( flu_array[total_sc] * ( sigma * sigma )* 2 * M_PI);
                                   }
                                if ( strcmp ( flattop , pshape ) == 0 )
                                   {
                                    flu_array[total_sc] = (nph * 0.8825) / ( sigma * sigma );        
                                    dnph = (flu_array[total_sc] * ( sigma * sigma ) ) / 0.8825 ;
                                   }                                          
           
                                if ( strcmp ( hybrid , pshape ) == 0 )
                                   {
                                    flu_array[total_sc] = (nph * 0.8825) / ( sigma * sigma );        
                                    dnph = (flu_array[total_sc] * ( sigma * sigma ) ) / 0.8825 ;
                                   }

                                ff1 = 0 ; ff2 = 5e60 ; ff3 = 0 ;
          
                                for ( r = 0 ; r < flu_count ; r++ )
                                   {
                                    ff1 = fabs(( flu_array[total_sc]/1e-8 ) - in_fluence[r] ) ;
                                    if ( ff1 < ff2 )
                                      {
                                         ff2 = ff1 ;
                                         ff3 = r ;
                                      }
                                   }
   
                                flu_array[total_sc] = in_fluence[ff3]*1e-8 ;
                                fprintf(fluenceout, "% le % le %le %le % le %le\n", 
                                                     i*scellx , j*scelly , k*scellz ,
                                                     flu_array[total_sc]/1e-8 , dnph,
                                                     ((flu_array[total_sc]/1e-8)-
                                                     (flu_array[total_sc-1]/1e-8))/(flu_array[total_sc-1]/1e-8));
          
                                sprintf (sflu, "%1.3e.", flu_array[total_sc]/1e-8);
	                        rl = yrandom (1, NRealiz+1) ;
                                sprintf (rliz, "R%07d", rl);
	                        strcat (strcat (strcat (src, sflu), rliz), snp);
	                        fprintf (info_out, "%s\n", src);
	                        end_dir[0] = '\0';
                                total_sc = total_sc + 1 ;
	                      }
                         }
                    }
         
             fclose (info_out);
             fclose (fluenceout);
             printf ("Total No. of SuperCell in focus    = % d\n",total_sc );

	     dir_info = fopen (info_log, "r");	// read only
             do
               {
                cc = fgetc (dir_info);
                if (cc == '\n')
                dd++;
               }
               while (cc != EOF);
               fclose (dir_info);



            CELL = calloc(total_sc * 3 , sizeof ( double ) );
    
           for ( i = -nsellx; i < nsellx + 1; i++ )
        	{ 
	        for ( j = -nselly; j < nselly + 1; j++ )
	            {
	            for ( k = 0; k < depth_cryst ; k++ )
	        	{
                         CELL[ijk_count][0] = i ;
                         CELL[ijk_count][1] = j ;
                         CELL[ijk_count][2] = k ;
                         ijk_count++ ;
                        }
                    }
                }
         }


          /////////////////////////////////////////////
	  ////////////////////////////////////////////
	  /////////////////////////////////////////////
          
     
          if ( strcmp( spi,sample_type ) == 0.0 ) 
               {
		   strcpy(flunce_info,"./output/flunce_info") ;
                   fluenceout = fopen (flunce_info,"w");
                   total_sc = NRealiz ;
                   CELL = calloc(NRealiz * 3 , sizeof ( double ) );
                if ( strcmp( flattop,pshape ) == 0 )
                   {
                    peak_flue = (nph * 0.8825 )  / (sigma * sigma) ; 
                   }
                
                
                printf("Peak Fluence (1/micron^2)          = % le\n", peak_flue/1e-8 );
                printf ("\n");
	        printf ("\n");
                for( i = 0 ;i <  NRealiz; i++)
	           {
		         flu_array[i] = in_fluence[0]*1e-8;
		         CELL[ijk_count][0] = 0 ;
                         CELL[ijk_count][1] = 0 ;
                         CELL[ijk_count][2] = 0 ;
			 
			 fprintf(fluenceout, "% le % le %le %le\n", CELL[ijk_count][0] , 
				 CELL[ijk_count][1] , CELL[ijk_count][2] , flu_array[i]/1e-8);
			 
			 
			 sprintf (sflu, "%1.3e.", flu_array[i]/1e-8);
			 sprintf (rliz, "R%07d", i+1);
			 strcat (strcat (strcat (src, sflu), rliz), snp);
	                 fprintf (info_out, "%s\n", src);
	                 end_dir[0] = '\0';
                         
			 ijk_count++ ;
	           }
          
                  fclose (info_out);
                  fclose (fluenceout);
                }

  ////////////////////////////////////////////////////////////////////
  //Calculating the number of different atomic species in the sample//
  ////////////////////////////////////////////////////////////////////
//                  printf("******Coordinate system in Angstroms for XTANT CALCULATIONS********\n\n");
		  char elements_info_char[1000],flu_part_string[1000];
		
		  strcpy (sample, ".R0000001/snp/00001000/Z.dat");    
		  strncpy(elements_info_char, src, sizeof (src));    
	      // src2[strlen(src2)-4]=0; This line works perfectly well I hope it will come handy in the future
		
		  sprintf (flu_part_string, "%1.3e", flu_array[0]/1e-8);
		  flu_part_string[sizeof (flu_part_string) - 1] = '\0';
		  strcat (strcat(elements_info_char,flu_part_string),sample);
		  
		  FILE *elements_info_file;
		  elements_info_file = fopen (elements_info_char, "r");
		  if (!elements_info_file)
                    {
                      printf ("Elements info is missing %s\n", ee);
                      exit (-1); 
                    }		  
		  
		  fclose (elements_info_file);
  
 
  
		  int *elements_in_sample;
		  elements_in_sample = calloc (N , sizeof (int));	                                
                  atomic_num_atoms(N  , elements_in_sample , elements_info_char) ;
                  
		  for (i = 0; i < N ; i++)
		  {
		    
		      if ( 1 == elements_in_sample[i]){Hydrogen_number_count = 	 	  Hydrogen_number_count + 1 ; } 
		      if ( 2 == elements_in_sample[i]){Helium_number_count = 	 	  Helium_number_count + 1 ; } 
		      if ( 3 == elements_in_sample[i]){Lithium_number_count = 	 	  Lithium_number_count + 1 ; } 
		      if ( 4 == elements_in_sample[i]){Beryllium_number_count = 	  Beryllium_number_count + 1 ; } 
		      if ( 5 == elements_in_sample[i]){Boron_number_count = 	          Boron_number_count + 1 ; } 
		      if ( 6 == elements_in_sample[i]){Carbon_number_count = 	  	  Carbon_number_count + 1 ; } 
		      if ( 7 == elements_in_sample[i]){Nitrogen_number_count = 	 	  Nitrogen_number_count + 1 ; } 
		      if ( 8 == elements_in_sample[i]){Oxygen_number_count = 	  	  Oxygen_number_count + 1 ; } 
		      if ( 9 == elements_in_sample[i]){Fluorine_number_count = 	  	  Fluorine_number_count + 1 ; } 
		      if ( 10 == elements_in_sample[i]){Neon_number_count = 	          Neon_number_count + 1 ; } 
		      if ( 11 == elements_in_sample[i]){Sodium_number_count = 	  	  Sodium_number_count + 1 ; } 
		      if ( 12 == elements_in_sample[i]){Magnesium_number_count = 	  Magnesium_number_count + 1 ; } 
		      if ( 13 == elements_in_sample[i]){Aluminum_number_count = 	  Aluminum_number_count + 1 ; } 
		      if ( 14 == elements_in_sample[i]){Silicon_number_count = 	  	  Silicon_number_count + 1 ; } 
		      if ( 15 == elements_in_sample[i]){ Phosphorus_number_count = 	  Phosphorus_number_count + 1 ; } 
		      if ( 16 == elements_in_sample[i]){Sulfur_number_count = 	  	  Sulfur_number_count + 1 ; } 
		      if ( 17 == elements_in_sample[i]){Chlorine_number_count = 	  Chlorine_number_count + 1 ; } 
		      if ( 18 == elements_in_sample[i]){Argon_number_count = 	          Argon_number_count + 1 ; } 
		      if ( 19 == elements_in_sample[i]){Potassium_number_count = 	  Potassium_number_count + 1 ; } 
		      if ( 20 == elements_in_sample[i]){Calcium_number_count = 	  	  Calcium_number_count + 1 ; } 
		      if ( 21 == elements_in_sample[i]){Scandium_number_count = 	  Scandium_number_count + 1 ; } 
		      if ( 22 == elements_in_sample[i]){Titanium_number_count = 	  Titanium_number_count + 1 ; } 
		      if ( 23 == elements_in_sample[i]){Vanadium_number_count = 	  Vanadium_number_count + 1 ; } 
		      if ( 24 == elements_in_sample[i]){Chromium_number_count = 	  Chromium_number_count + 1 ; } 
		      if ( 25 == elements_in_sample[i]){Manganese_number_count = 	  Manganese_number_count + 1 ; } 
		      if ( 26 == elements_in_sample[i]){Iron_number_count = 	          Iron_number_count + 1 ; } 
		      if ( 27 == elements_in_sample[i]){Cobalt_number_count = 	 	  Cobalt_number_count + 1 ; } 
		      if ( 28 == elements_in_sample[i]){Nickel_number_count = 	 	  Nickel_number_count + 1 ; } 
		      if ( 29 == elements_in_sample[i]){Copper_number_count = 	 	  Copper_number_count + 1 ; } 
		      if ( 30 == elements_in_sample[i]){Zinc_number_count = 	          Zinc_number_count + 1 ; } 
		      if ( 31 == elements_in_sample[i]){Gallium_number_count = 	 	  Gallium_number_count + 1 ; } 
		      if ( 32 == elements_in_sample[i]){Germanium_number_count = 	  Germanium_number_count + 1 ; } 
		      if ( 33 == elements_in_sample[i]){Arsenic_number_count = 	 	  Arsenic_number_count + 1 ; } 
		      if ( 34 == elements_in_sample[i]){Selenium_number_count = 	  Selenium_number_count + 1 ; } 
		      if ( 35 == elements_in_sample[i]){Bromine_number_count = 	 	  Bromine_number_count + 1 ; } 
		      if ( 36 == elements_in_sample[i]){Krypton_number_count = 	    	  Krypton_number_count + 1 ; } 
		      if ( 37 == elements_in_sample[i]){Rubidium_number_count = 	  Rubidium_number_count + 1 ; } 
		      if ( 38 == elements_in_sample[i]){Strontium_number_count = 	  Strontium_number_count + 1 ; } 
		      if ( 39 == elements_in_sample[i]){Yttrium_number_count = 	  	  Yttrium_number_count + 1 ; } 
		      if ( 40 == elements_in_sample[i]){Zirconium_number_count = 	  Zirconium_number_count + 1 ; } 
		      if ( 41 == elements_in_sample[i]){ Niobium_number_count = 	  Niobium_number_count + 1 ; } 
		      if ( 42 == elements_in_sample[i]){Molybdenum_number_count = 	  Molybdenum_number_count + 1 ; } 
		      if ( 43 == elements_in_sample[i]){Technetium_number_count = 	  Technetium_number_count + 1 ; } 
		      if ( 44 == elements_in_sample[i]){Ruthenium_number_count = 	  Ruthenium_number_count + 1 ; } 
		      if ( 45 == elements_in_sample[i]){Rhodium_number_count = 	  	  Rhodium_number_count + 1 ; } 
		      if ( 46 == elements_in_sample[i]){Palladium_number_count = 	  Palladium_number_count + 1 ; } 
		      if ( 47 == elements_in_sample[i]){Silver_number_count = 	  	  Silver_number_count + 1 ; } 
		      if ( 48 == elements_in_sample[i]){Cadmium_number_count = 	  	  Cadmium_number_count + 1 ; } 
		      if ( 49 == elements_in_sample[i]){Indium_number_count = 	 	  Indium_number_count + 1 ; } 
		      if ( 50 == elements_in_sample[i]){Tin_number_count = 	 	  Tin_number_count + 1 ; } 
		      if ( 51 == elements_in_sample[i]){Antimony_number_count = 	  Antimony_number_count + 1 ; } 
		      if ( 52 == elements_in_sample[i]){Tellurium_number_count = 	  Tellurium_number_count + 1 ; } 
		      if ( 53 == elements_in_sample[i]){ Iodine_number_count = 	  	  Iodine_number_count + 1 ; } 
		      if ( 54 == elements_in_sample[i]){ Xenon_number_count = 	 	  Xenon_number_count + 1 ; } 
		      if ( 55 == elements_in_sample[i]){Cesium_number_count = 	 	  Cesium_number_count + 1 ; } 
		      if ( 56 == elements_in_sample[i]){Barium_number_count = 	   	  Barium_number_count + 1 ; } 
		      if ( 57 == elements_in_sample[i]){Lanthanum_number_count = 	  Lanthanum_number_count + 1 ; } 
		      if ( 58 == elements_in_sample[i]){Cerium_number_count = 	  	  Cerium_number_count + 1 ; } 
		      if ( 59 == elements_in_sample[i]){Praseodymium_number_count = 	  Praseodymium_number_count + 1 ; } 
		      if ( 60 == elements_in_sample[i]){Neodymium_number_count = 	  Neodymium_number_count + 1 ; } 
		      if ( 61 == elements_in_sample[i]){Promethium_number_count = 	  Promethium_number_count + 1 ; } 
		      if ( 62 == elements_in_sample[i]){Samarium_number_count = 	  Samarium_number_count + 1 ; } 
		      if ( 63 == elements_in_sample[i]){Europium_number_count = 	  Europium_number_count + 1 ; } 
		      if ( 64 == elements_in_sample[i]){Gadolinium_number_count = 	  Gadolinium_number_count + 1 ; } 
		      if ( 65 == elements_in_sample[i]){Terbium_number_count = 	  	  Terbium_number_count + 1 ; } 
		      if ( 66 == elements_in_sample[i]){Dysprosium_number_count = 	  Dysprosium_number_count + 1 ; } 
		      if ( 67 == elements_in_sample[i]){Holmium_number_count = 	  	  Holmium_number_count + 1 ; } 
		      if ( 68 == elements_in_sample[i]){ Erbium_number_count = 	  	  Erbium_number_count + 1 ; } 
		      if ( 69 == elements_in_sample[i]){Thulium_number_count = 	  	  Thulium_number_count + 1 ; } 
		      if ( 70 == elements_in_sample[i]){Ytterbium_number_count = 	  Ytterbium_number_count + 1 ; } 
		      if ( 71 == elements_in_sample[i]){Lutetium_number_count = 	  Lutetium_number_count + 1 ; } 
		      if ( 72 == elements_in_sample[i]){Hafnium_number_count = 	  	  Hafnium_number_count + 1 ; } 
		      if ( 73 == elements_in_sample[i]){ Tantalum_number_count = 	  Tantalum_number_count + 1 ; } 
		      if ( 74 == elements_in_sample[i]){Tungsten_number_count = 	  Tungsten_number_count + 1 ; } 
		      if ( 75 == elements_in_sample[i]){Rhenium_number_count = 	 	  Rhenium_number_count + 1 ; } 
		      if ( 76 == elements_in_sample[i]){Osmium_number_count = 	  	  Osmium_number_count + 1 ; } 
		      if ( 77 == elements_in_sample[i]){Iridium_number_count = 	  	  Iridium_number_count + 1 ; } 
		      if ( 78 == elements_in_sample[i]){Platinum_number_count = 	  Platinum_number_count + 1 ; } 
		      if ( 79 == elements_in_sample[i]){Gold_number_count = 	 	  Gold_number_count + 1 ; } 
		      if ( 80 == elements_in_sample[i]){ Mercury_number_count = 	  Mercury_number_count + 1 ; } 
		      if ( 81 == elements_in_sample[i]){Thallium_number_count = 	  Thallium_number_count + 1 ; } 
		      if ( 82 == elements_in_sample[i]){Lead_number_count = 	  	  Lead_number_count + 1 ; } 
		      if ( 83 == elements_in_sample[i]){Bismuth_number_count = 	 	  Bismuth_number_count + 1 ; } 
		      if ( 84 == elements_in_sample[i]){Polonium_number_count = 	  Polonium_number_count + 1 ; } 
		      if ( 85 == elements_in_sample[i]){Astatine_number_count = 	  Astatine_number_count + 1 ; } 
		      if ( 86 == elements_in_sample[i]){Radon_number_count = 	 	  Radon_number_count + 1 ; } 
		      if ( 87 == elements_in_sample[i]){Francium_number_count = 	  Francium_number_count + 1 ; } 
		      if ( 88 == elements_in_sample[i]){Radium_number_count = 	 	  Radium_number_count + 1 ; } 
		      if ( 89 == elements_in_sample[i]){Actinium_number_count = 	  Actinium_number_count + 1 ; } 
		      if ( 90 == elements_in_sample[i]){Thorium_number_count = 	 	  Thorium_number_count + 1 ; } 
		      if ( 91 == elements_in_sample[i]){Protactinium_number_count = 	  Protactinium_number_count + 1 ; } 
		      if ( 92 == elements_in_sample[i]){Uranium_number_count = 	 	  Uranium_number_count + 1 ; } 
		      if ( 93 == elements_in_sample[i]){Neptunium_number_count = 	  Neptunium_number_count + 1 ; } 
		      if ( 94 == elements_in_sample[i]){Plutonium_number_count = 	  Plutonium_number_count + 1 ; } 
		      if ( 95 == elements_in_sample[i]){Americium_number_count = 	  Americium_number_count + 1 ; } 
		      if ( 96 == elements_in_sample[i]){Curium_number_count = 	  	  Curium_number_count + 1 ; } 
		      if ( 97 == elements_in_sample[i]){Berkelium_number_count = 	  Berkelium_number_count + 1 ; } 
		      if ( 98 == elements_in_sample[i]){Californium_number_count = 	  Californium_number_count + 1 ; } 
		      if ( 99 == elements_in_sample[i]){Einsteinium_number_count = 	  Einsteinium_number_count + 1 ; } 
		      if ( 100 == elements_in_sample[i]){Fermium_number_count = 	  Fermium_number_count + 1 ; } 
		      if ( 101 == elements_in_sample[i]){Mendelevium_number_count = 	  Mendelevium_number_count + 1 ; } 
		      if ( 102 == elements_in_sample[i]){Nobelium_number_count = 	  Nobelium_number_count + 1 ; } 
		      if ( 103 == elements_in_sample[i]){Lawrencium_number_count = 	  Lawrencium_number_count + 1 ; } 
		      if ( 104 == elements_in_sample[i]){Rutherfordium_number_count = 	  Rutherfordium_number_count + 1 ; } 
		      if ( 105 == elements_in_sample[i]){Dubnium_number_count = 	  Dubnium_number_count + 1 ; } 
		      if ( 106 == elements_in_sample[i]){Seaborgium_number_count = 	  Seaborgium_number_count + 1 ; } 
		      if ( 107 == elements_in_sample[i]){Bohrium_number_count = 	  Bohrium_number_count + 1 ; } 
		      if ( 108 == elements_in_sample[i]){Hassium_number_count = 	  Hassium_number_count + 1 ; } 
		      if ( 109 == elements_in_sample[i]){Meitnerium_number_count = 	  Meitnerium_number_count + 1 ; } 
					
		  }
		  
		  
                  free(elements_in_sample);
  

            
         
  ////////////////////////////////////////////////////////////////////  
  /////////////////Writting the Input parameters//////////////////////  
  ////////////////////////////////////////////////////////////////////  
  
  strcpy(input_info,"./output/input_info") ;
  inputout = fopen (input_info,"w");
  fprintf(inputout,"***********************************************************\n");
  fprintf(inputout,"INFORMATION ABOUT THE XSINC SCATTERING INTENSITY PARAMETERS\n");
  fprintf(inputout,"***********************************************************\n\n");

  fprintf (inputout,"\n");
  fprintf (inputout,"Xsinc Revision on SVN              = % s\n", SVN_VERSION);
  fprintf (inputout,"Type Of the Sample                 = % s\n", sample_type);
  fprintf (inputout,"Type Of the System                 = % s\n", system_type); 
  fprintf (inputout,"F Double Prime                     = % s\n", fpp_status ); 
  fprintf (inputout,"No. OF THREADS                     = % d\n", num_threads); 
  fprintf (inputout,"INPUT DATA PATH                    = % s\n", src);
  fprintf (inputout,"PULSE SHAPE                        = % s\n", pshape);
  fprintf (inputout,"TEMPORAL PULSE SHAPE               = % s\n", temporal_pulse);
  fprintf (inputout,"EffectiveF                         = % s\n ", effec_f);
  fprintf (inputout,"SIGMA (Angstroms)                  = % 1f\n", sigma);
  fprintf (inputout,"No. of Particles per Super Cell    = % d\n", N);
  fprintf (inputout,"No. of the Shells of a Crystal     = % d %d\n", nsellx, nselly);
  fprintf (inputout,"Layers of the Depth of the Crystal = % d\n", depth_cryst);
  fprintf (inputout,"SUPERCELL SIZE (Angstrom)          = % 1f % 1f % 1f\n", scellx, scelly, scellz);
  fprintf (inputout,"QPOINTS                            = % d\n", qp_rcount );
  fprintf (inputout,"No OF SINGLE FLUENCE REALIZATIONS  = % d\n", NRealiz );
  fprintf (inputout,"TOTAL TIMESTEPS                    = % le\n", FinalSnap/sp_diff);
  fprintf (inputout,"PULSE DURATION                     = % .0f fs\n", tau/1000);
  fprintf (inputout,"CENTRE OF PULSE                    = % .0f fs\n", T0/1000);
  fprintf (inputout,"Peak Fluence (1/micron^2)          = % le\n", peak_flue/1e-8 );
  fprintf (inputout,"\n");
  fprintf (inputout,"*********************************************************\n");
  fprintf (inputout,"INFORMATION ABOUT THE XSINC MOLECULAR DYNAMICS PARAMETERS\n");
  fprintf (inputout,"*********************************************************\n\n");
  fprintf (inputout,"\n");
  fprintf (inputout,"Molecular Dynamics                 = % s\n", MD);
  fprintf (inputout,"regparam                           = % le\n", regparam);
  fprintf (inputout,"alpha                              = % le\n", alphaforewald);
  fprintf (inputout,"Rcut                               = % d\n", rcut);
  fprintf (inputout,"Kcut                               = % d\n", kcut);
  fprintf (inputout,"MinImg                             = % s\n", min_img);
  fprintf (inputout,"HARDWALL                           = % s\n", hard_wall);
  fprintf (inputout,"RANDOM SEED                        = % d\n", &SEED);

  fclose (inputout);
  
  ////////////////////////////////////////////////////////////////////
  /////////////////Starting the Main Program here/////////////////////  
  ////////////////////////////////////////////////////////////////////
      
     int str_num  , verify_num ;
     double chk_xxx;
     char (*chk_string)[500];
     chk_string = calloc (total_sc * 500 , sizeof (char));
     char copy_var[500];
     int copy_var_count , xxxx=0 ;
     int type_check  ,fpp_check , dummy_ts = 0; 
     chk_xxx = flu_count * NRealiz ;
  //   double complex (*intensity_account)[chk_xxx];
     double complex (*intensity_account)[qp_rcount];
  //   double complex (*qrterm_account)[chk_xxx];
     double complex (*qrterm_account)[qp_rcount];

double (*Hydrogen_form_factor_account)[qp_rcount];
double (*Helium_form_factor_account)[qp_rcount];
double (*Lithium_form_factor_account)[qp_rcount];
double (*Beryllium_form_factor_account)[qp_rcount];
double (*Boron_form_factor_account)[qp_rcount];
double (*Carbon_form_factor_account)[qp_rcount];
double (*Nitrogen_form_factor_account)[qp_rcount];
double (*Oxygen_form_factor_account)[qp_rcount];
double (*Fluorine_form_factor_account)[qp_rcount];
double (*Neon_form_factor_account)[qp_rcount];
double (*Sodium_form_factor_account)[qp_rcount];
double (*Magnesium_form_factor_account)[qp_rcount];
double (*Aluminum_form_factor_account)[qp_rcount];
double (*Silicon_form_factor_account)[qp_rcount];
double (*Phosphorus_form_factor_account)[qp_rcount];
double (*Sulfur_form_factor_account)[qp_rcount];
double (*Chlorine_form_factor_account)[qp_rcount];
double (*Argon_form_factor_account)[qp_rcount];
double (*Potassium_form_factor_account)[qp_rcount];
double (*Calcium_form_factor_account)[qp_rcount];
double (*Scandium_form_factor_account)[qp_rcount];
double (*Titanium_form_factor_account)[qp_rcount];
double (*Vanadium_form_factor_account)[qp_rcount];
double (*Chromium_form_factor_account)[qp_rcount];
double (*Manganese_form_factor_account)[qp_rcount];
double (*Iron_form_factor_account)[qp_rcount];
double (*Cobalt_form_factor_account)[qp_rcount];
double (*Nickel_form_factor_account)[qp_rcount];
double (*Copper_form_factor_account)[qp_rcount];
double (*Zinc_form_factor_account)[qp_rcount];
double (*Gallium_form_factor_account)[qp_rcount];
double (*Germanium_form_factor_account)[qp_rcount];
double (*Arsenic_form_factor_account)[qp_rcount];
double (*Selenium_form_factor_account)[qp_rcount];
double (*Bromine_form_factor_account)[qp_rcount];
double (*Krypton_form_factor_account)[qp_rcount];
double (*Rubidium_form_factor_account)[qp_rcount];
double (*Strontium_form_factor_account)[qp_rcount];
double (*Yttrium_form_factor_account)[qp_rcount];
double (*Zirconium_form_factor_account)[qp_rcount];
double (*Niobium_form_factor_account)[qp_rcount];
double (*Molybdenum_form_factor_account)[qp_rcount];
double (*Technetium_form_factor_account)[qp_rcount];
double (*Ruthenium_form_factor_account)[qp_rcount];
double (*Rhodium_form_factor_account)[qp_rcount];
double (*Palladium_form_factor_account)[qp_rcount];
double (*Silver_form_factor_account)[qp_rcount];
double (*Cadmium_form_factor_account)[qp_rcount];
double (*Indium_form_factor_account)[qp_rcount];
double (*Tin_form_factor_account)[qp_rcount];
double (*Antimony_form_factor_account)[qp_rcount];
double (*Tellurium_form_factor_account)[qp_rcount];
double (*Iodine_form_factor_account)[qp_rcount];
double (*Xenon_form_factor_account)[qp_rcount];
double (*Cesium_form_factor_account)[qp_rcount];
double (*Barium_form_factor_account)[qp_rcount];
double (*Lanthanum_form_factor_account)[qp_rcount];
double (*Cerium_form_factor_account)[qp_rcount];
double (*Praseodymium_form_factor_account)[qp_rcount];
double (*Neodymium_form_factor_account)[qp_rcount];
double (*Promethium_form_factor_account)[qp_rcount];
double (*Samarium_form_factor_account)[qp_rcount];
double (*Europium_form_factor_account)[qp_rcount];
double (*Gadolinium_form_factor_account)[qp_rcount];
double (*Terbium_form_factor_account)[qp_rcount];
double (*Dysprosium_form_factor_account)[qp_rcount];
double (*Holmium_form_factor_account)[qp_rcount];
double (*Erbium_form_factor_account)[qp_rcount];
double (*Thulium_form_factor_account)[qp_rcount];
double (*Ytterbium_form_factor_account)[qp_rcount];
double (*Lutetium_form_factor_account)[qp_rcount];
double (*Hafnium_form_factor_account)[qp_rcount];
double (*Tantalum_form_factor_account)[qp_rcount];
double (*Tungsten_form_factor_account)[qp_rcount];
double (*Rhenium_form_factor_account)[qp_rcount];
double (*Osmium_form_factor_account)[qp_rcount];
double (*Iridium_form_factor_account)[qp_rcount];
double (*Platinum_form_factor_account)[qp_rcount];
double (*Gold_form_factor_account)[qp_rcount];
double (*Mercury_form_factor_account)[qp_rcount];
double (*Thallium_form_factor_account)[qp_rcount];
double (*Lead_form_factor_account)[qp_rcount];
double (*Bismuth_form_factor_account)[qp_rcount];
double (*Polonium_form_factor_account)[qp_rcount];
double (*Astatine_form_factor_account)[qp_rcount];
double (*Radon_form_factor_account)[qp_rcount];
double (*Francium_form_factor_account)[qp_rcount];
double (*Radium_form_factor_account)[qp_rcount];
double (*Actinium_form_factor_account)[qp_rcount];
double (*Thorium_form_factor_account)[qp_rcount];
double (*Protactinium_form_factor_account)[qp_rcount];
double (*Uranium_form_factor_account)[qp_rcount];
double (*Neptunium_form_factor_account)[qp_rcount];
double (*Plutonium_form_factor_account)[qp_rcount];
double (*Americium_form_factor_account)[qp_rcount];
double (*Curium_form_factor_account)[qp_rcount];
double (*Berkelium_form_factor_account)[qp_rcount];
double (*Californium_form_factor_account)[qp_rcount];
double (*Einsteinium_form_factor_account)[qp_rcount];
double (*Fermium_form_factor_account)[qp_rcount];
double (*Mendelevium_form_factor_account)[qp_rcount];
double (*Nobelium_form_factor_account)[qp_rcount];
double (*Lawrencium_form_factor_account)[qp_rcount];
double (*Rutherfordium_form_factor_account)[qp_rcount];
double (*Dubnium_form_factor_account)[qp_rcount];
double (*Seaborgium_form_factor_account)[qp_rcount];
double (*Bohrium_form_factor_account)[qp_rcount];
double (*Hassium_form_factor_account)[qp_rcount];
double (*Meitnerium_form_factor_account)[qp_rcount];

    
    
    if (strcmp ( damaged,system_type ) == 0)
       {
           type_check = 0 ;
       }
  
    if (strcmp ( undamaged,system_type ) == 0)
       {
           type_check = 1 ;
       }
    
    if (strcmp ( on,fpp_status ) == 0)
       {
           fpp_check = 1 ;
       }

    if (strcmp ( off,fpp_status ) == 0)
       {
           fpp_check = 0 ;
       }
     
   for ( ts = st_snp; ts <= FinalSnap; ts = ts + sp_diff )
      {
        printf("              TIMESTEP %.1f \n",ts/(double)1000);
     //   intensity_account = calloc (qp_rcount * chk_xxx , sizeof (double complex));
        intensity_account = calloc (chk_xxx * qp_rcount , sizeof (double complex));
     //   qrterm_account = calloc (qp_rcount * chk_xxx , sizeof (double complex));
        qrterm_account = calloc (chk_xxx * qp_rcount , sizeof (double complex));
        
	
	Hydrogen_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double)); 
	Helium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Lithium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Beryllium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Boron_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Carbon_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Nitrogen_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Oxygen_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Fluorine_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Neon_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Sodium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Magnesium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Aluminum_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Silicon_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Phosphorus_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Sulfur_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Chlorine_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Argon_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Potassium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Calcium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Scandium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Titanium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Vanadium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Chromium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Manganese_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Iron_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Cobalt_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Nickel_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Copper_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Zinc_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Gallium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Germanium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Arsenic_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Selenium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Bromine_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Krypton_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Rubidium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Strontium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Yttrium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Zirconium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Niobium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Molybdenum_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Technetium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Ruthenium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Rhodium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Palladium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Silver_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Cadmium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Indium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Tin_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Antimony_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Tellurium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Iodine_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Xenon_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Cesium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Barium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Lanthanum_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Cerium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Praseodymium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Neodymium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Promethium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Samarium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Europium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Gadolinium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Terbium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Dysprosium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Holmium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Erbium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Thulium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Ytterbium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Lutetium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Hafnium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Tantalum_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Tungsten_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Rhenium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Osmium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Iridium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Platinum_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Gold_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Mercury_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Thallium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Lead_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Bismuth_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Polonium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Astatine_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Radon_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Francium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Radium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Actinium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Thorium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Protactinium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Uranium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Neptunium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Plutonium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Americium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Curium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Berkelium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Californium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Einsteinium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Fermium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Mendelevium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Nobelium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Lawrencium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Rutherfordium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Dubnium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Seaborgium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Bohrium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Hassium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
	Meitnerium_form_factor_account = calloc (chk_xxx * qp_rcount , sizeof (double));
		
		
		
	Hydrogen_form_factor = calloc (qp_rcount , sizeof (double));
	Helium_form_factor = calloc (qp_rcount , sizeof (double));
	Lithium_form_factor = calloc (qp_rcount , sizeof (double));
	Beryllium_form_factor = calloc (qp_rcount , sizeof (double));
	Boron_form_factor = calloc (qp_rcount , sizeof (double));
	Carbon_form_factor = calloc (qp_rcount , sizeof (double));
	Nitrogen_form_factor = calloc (qp_rcount , sizeof (double));
	Oxygen_form_factor = calloc (qp_rcount , sizeof (double));
	Fluorine_form_factor = calloc (qp_rcount , sizeof (double));
	Neon_form_factor = calloc (qp_rcount , sizeof (double));
	Sodium_form_factor = calloc (qp_rcount , sizeof (double));
	Magnesium_form_factor = calloc (qp_rcount , sizeof (double));
	Aluminum_form_factor = calloc (qp_rcount , sizeof (double));
	Silicon_form_factor = calloc (qp_rcount , sizeof (double));
	Phosphorus_form_factor = calloc (qp_rcount , sizeof (double));
	Sulfur_form_factor = calloc (qp_rcount , sizeof (double));
	Chlorine_form_factor = calloc (qp_rcount , sizeof (double));
	Argon_form_factor = calloc (qp_rcount , sizeof (double));
	Potassium_form_factor = calloc (qp_rcount , sizeof (double));
	Calcium_form_factor = calloc (qp_rcount , sizeof (double));
	Scandium_form_factor = calloc (qp_rcount , sizeof (double));
	Titanium_form_factor = calloc (qp_rcount , sizeof (double));
	Vanadium_form_factor = calloc (qp_rcount , sizeof (double));
	Chromium_form_factor = calloc (qp_rcount , sizeof (double));
	Manganese_form_factor = calloc (qp_rcount , sizeof (double));
	Iron_form_factor = calloc (qp_rcount , sizeof (double));
	Cobalt_form_factor = calloc (qp_rcount , sizeof (double));
	Nickel_form_factor = calloc (qp_rcount , sizeof (double));
	Copper_form_factor = calloc (qp_rcount , sizeof (double));
	Zinc_form_factor = calloc (qp_rcount , sizeof (double));
	Gallium_form_factor = calloc (qp_rcount , sizeof (double));
	Germanium_form_factor = calloc (qp_rcount , sizeof (double));
	Arsenic_form_factor = calloc (qp_rcount , sizeof (double));
	Selenium_form_factor = calloc (qp_rcount , sizeof (double));
	Bromine_form_factor = calloc (qp_rcount , sizeof (double));
	Krypton_form_factor = calloc (qp_rcount , sizeof (double));
	Rubidium_form_factor = calloc (qp_rcount , sizeof (double));
	Strontium_form_factor = calloc (qp_rcount , sizeof (double));
	Yttrium_form_factor = calloc (qp_rcount , sizeof (double));
	Zirconium_form_factor = calloc (qp_rcount , sizeof (double));
	Niobium_form_factor = calloc (qp_rcount , sizeof (double));
	Molybdenum_form_factor = calloc (qp_rcount , sizeof (double));
	Technetium_form_factor = calloc (qp_rcount , sizeof (double));
	Ruthenium_form_factor = calloc (qp_rcount , sizeof (double));
	Rhodium_form_factor = calloc (qp_rcount , sizeof (double));
	Palladium_form_factor = calloc (qp_rcount , sizeof (double));
	Silver_form_factor = calloc (qp_rcount , sizeof (double));
	Cadmium_form_factor = calloc (qp_rcount , sizeof (double));
	Indium_form_factor = calloc (qp_rcount , sizeof (double));
	Tin_form_factor = calloc (qp_rcount , sizeof (double));
	Antimony_form_factor = calloc (qp_rcount , sizeof (double));
	Tellurium_form_factor = calloc (qp_rcount , sizeof (double));
	Iodine_form_factor = calloc (qp_rcount , sizeof (double));
	Xenon_form_factor = calloc (qp_rcount , sizeof (double));
	Cesium_form_factor = calloc (qp_rcount , sizeof (double));
	Barium_form_factor = calloc (qp_rcount , sizeof (double));
	Lanthanum_form_factor = calloc (qp_rcount , sizeof (double));
	Cerium_form_factor = calloc (qp_rcount , sizeof (double));
	Praseodymium_form_factor = calloc (qp_rcount , sizeof (double));
	Neodymium_form_factor = calloc (qp_rcount , sizeof (double));
	Promethium_form_factor = calloc (qp_rcount , sizeof (double));
	Samarium_form_factor = calloc (qp_rcount , sizeof (double));
	Europium_form_factor = calloc (qp_rcount , sizeof (double));
	Gadolinium_form_factor = calloc (qp_rcount , sizeof (double));
	Terbium_form_factor = calloc (qp_rcount , sizeof (double));
	Dysprosium_form_factor = calloc (qp_rcount , sizeof (double));
	Holmium_form_factor = calloc (qp_rcount , sizeof (double));
	Erbium_form_factor = calloc (qp_rcount , sizeof (double));
	Thulium_form_factor = calloc (qp_rcount , sizeof (double));
	Ytterbium_form_factor = calloc (qp_rcount , sizeof (double));
	Lutetium_form_factor = calloc (qp_rcount , sizeof (double));
	Hafnium_form_factor = calloc (qp_rcount , sizeof (double));
	Tantalum_form_factor = calloc (qp_rcount , sizeof (double));
	Tungsten_form_factor = calloc (qp_rcount , sizeof (double));
	Rhenium_form_factor = calloc (qp_rcount , sizeof (double));
	Osmium_form_factor = calloc (qp_rcount , sizeof (double));
	Iridium_form_factor = calloc (qp_rcount , sizeof (double));
	Platinum_form_factor = calloc (qp_rcount , sizeof (double));
	Gold_form_factor = calloc (qp_rcount , sizeof (double));
	Mercury_form_factor = calloc (qp_rcount , sizeof (double));
	Thallium_form_factor = calloc (qp_rcount , sizeof (double));
	Lead_form_factor = calloc (qp_rcount , sizeof (double));
	Bismuth_form_factor = calloc (qp_rcount , sizeof (double));
	Polonium_form_factor = calloc (qp_rcount , sizeof (double));
	Astatine_form_factor = calloc (qp_rcount , sizeof (double));
	Radon_form_factor = calloc (qp_rcount , sizeof (double));
	Francium_form_factor = calloc (qp_rcount , sizeof (double));
	Radium_form_factor = calloc (qp_rcount , sizeof (double));
	Actinium_form_factor = calloc (qp_rcount , sizeof (double));
	Thorium_form_factor = calloc (qp_rcount , sizeof (double));
	Protactinium_form_factor = calloc (qp_rcount , sizeof (double));
	Uranium_form_factor = calloc (qp_rcount , sizeof (double));
	Neptunium_form_factor = calloc (qp_rcount , sizeof (double));
	Plutonium_form_factor = calloc (qp_rcount , sizeof (double));
	Americium_form_factor = calloc (qp_rcount , sizeof (double));
	Curium_form_factor = calloc (qp_rcount , sizeof (double));
	Berkelium_form_factor = calloc (qp_rcount , sizeof (double));
	Californium_form_factor = calloc (qp_rcount , sizeof (double));
	Einsteinium_form_factor = calloc (qp_rcount , sizeof (double));
	Fermium_form_factor = calloc (qp_rcount , sizeof (double));
	Mendelevium_form_factor = calloc (qp_rcount , sizeof (double));
	Nobelium_form_factor = calloc (qp_rcount , sizeof (double));
	Lawrencium_form_factor = calloc (qp_rcount , sizeof (double));
	Rutherfordium_form_factor = calloc (qp_rcount , sizeof (double));
	Dubnium_form_factor = calloc (qp_rcount , sizeof (double));
	Seaborgium_form_factor = calloc (qp_rcount , sizeof (double));
	Bohrium_form_factor = calloc (qp_rcount , sizeof (double));
	Hassium_form_factor = calloc (qp_rcount , sizeof (double));
	Meitnerium_form_factor = calloc (qp_rcount , sizeof (double));
		    
        f_0 = calloc (qp_rcount , sizeof (double complex));
        qrterm_comp = calloc (qp_rcount , sizeof (double complex));
        total_sc = 0 ; 
        sprintf (ts_out, "%08d", ts);
       
        strcat(output,ts_out) ;
        FILE *complex_out = fopen (output, "w");
        OP_dir[0] = '\0';
       
        strcat(coord_log, ts_out);
        FILE *coordinates_out = fopen (coord_log, "w");
        r_dir[0] = '\0';
       
        strcat(formfactor_log, ts_out);
        FILE *formfactor_out = fopen (formfactor_log, "w");
        ff_dir[0] = '\0';

        FILE *r_info = fopen (info_log, "r");	        // read only
        FILE *T_info = fopen (info_log, "r");	        // read only
        FILE *Q_info = fopen (info_log, "r");	        // read only
        FILE *f0_info = fopen (info_log, "r");	        // read only
        FILE *fpp_info = fopen (info_log, "r");  	// read only
        FILE *ele_r_info = fopen (info_log, "r");	// read only
        FILE *charge_info = fopen (info_log, "r");	// read only
        FILE *atomic_no_info = fopen (info_log, "r");	// read only
        copy_var_count = 0 ;
              
              for ( i = 0 ; i < ijk_count ; i++ )
                 {
		  ////////////////////////////////////////////////////////////////////  
		  /////////////////Verification of the Used FLUENCE///////////////////  
		  ////////////////////////////////////////////////////////////////////
		  comp_num = 0;
		  fscanf (r_info, "%s\n", ee);
		  if (type_check == 0)
                    {
                     sprintf (tstep, "%08d/", ts);
                     strcat (strcat (ee, tstep), rdat);
                    }
		  if (type_check == 1)
                    {
                     sprintf (tstep, "%08d/", dummy_ts);
                     strcat (strcat (ee, tstep), rdat);
                    }

                  copy_var[0] = '\0'; 
                  strcpy(copy_var,ee);
                  verify_num = 0 ;
        /*************************************************************************************************/
        /*************************************************************************************************/
        /*************************************************************************************************/
                 if(0 == i) 
                    {
                     strcpy(chk_string[copy_var_count] , copy_var ) ;
                  ////////////////////////////////////////////////////////////////////  
		  /////////////////Input Coordinates from Input File//////////////////  
		  ////////////////////////////////////////////////////////////////////
                  double (*cq)[3];
                  cq = calloc (N * 3, sizeof (double));	/* Exporting Coordinates Array */
                  coordinates_atoms( N , cq , ee) ;

		  ////////////////////////////////////////////////////////////////////  
		  ////////////////Input Species Type from Input File//////////////////  
		  ////////////////////////////////////////////////////////////////////
                  int *atyp ;
                  atyp = calloc (N, sizeof (int));	/* Exporting Charges Array */
		  
                  
                   if (strcmp(on,effec_f) == 0.0)
                      {
                        strcpy(ee_atypmod,"./atyp_mod.dat");
                        species_atoms_mod( N , atyp , ee_atypmod);
                 //     printf ("modified atom type path % s\n",ee_atypmod);
                      }
                  
                   if (strcmp(off,effec_f) == 0.0)
                      {
                        fscanf (T_info, "%s\n", ee);

		        if (type_check == 0)
                           {
                            sprintf (tstep, "%08d/", ts);
                            strcat (strcat (ee, tstep), tdat);
                           }
		        if (type_check == 1)
                           {
                            sprintf (tstep, "%08d/", dummy_ts);
                            strcat (strcat (ee, tstep), tdat);
                           }
                        species_atoms( N , atyp , ee) ;
                      }

                 //     int tss = 0 ;
                 //     int sumss = 0 ; 
                 //     for(tss ; tss < N ; tss++)
                 //         {
                 //             sumss = sumss + atyp[tss] ;
                              
                 //         }
                 //         printf("atomic type % d \n" , sumss);
		  ////////////////////////////////////////////////////////////////////  
		  //////////////Scanning f0 and f0_mod from the Input File////////////  
		  ////////////////////////////////////////////////////////////////////     

		  int ch;
		  int rcount = 0;
		  int lcount = 0;
                 
                 if (strcmp(on,effec_f) == 0.0)
                    {
                   //  fscanf (f0_modified, "%s\n", ee);
                       strcpy(ee_f0mod,"./f0_mod.dat");
		       FILE *fp = fopen (ee_f0mod, "r");
		       do
		        {
		         ch = fgetc (fp);
		         if (ch == '\n')
			 rcount++;
		        }
		       while (ch != EOF);
		       fclose (fp);


		       fp = fopen (ee_f0mod, "r");
		       while (0 == fscanf (fp, "%*s"))
		       {
		         lcount++;
		       }
		       fclose (fp);

		       lcount = lcount / rcount;
                    }
                    
                 if (strcmp(off,effec_f) == 0.0)
                    {
		          fscanf (f0_info, "%s\n", ee);
            
                          if (type_check == 0)
                              {
                               sprintf (tstep, "%08d/", ts);
                               strcat (strcat (ee, tstep), f0dat);
                               }
		          if (type_check == 1)
                               {
                               sprintf (tstep, "%08d/", dummy_ts);
                               strcat (strcat (ee, tstep), f0dat);
                               }
              		  FILE *fp = fopen (ee, "r");
	         	  do
		           {
		            ch = fgetc (fp);
		            if (ch == '\n')
			    rcount++;
		           }
		          while (ch != EOF);
		          fclose (fp);


		          fp = fopen (ee, "r");
		          while (0 == fscanf (fp, "%*s"))
		          {
		           lcount++;
		          }
		          fclose (fp);

		          lcount = lcount / rcount;
                    }

		  double (*f0)[lcount];
		  f0 = calloc (rcount * lcount, sizeof (double));	/* Exporting Charges Array */

		  ////////////////////////////////////////////////////////////////////  
		  ////////////////Input f0 $ f0_mod from Input File///////////////////  
		  ////////////////////////////////////////////////////////////////////

                 if (strcmp(off,effec_f) == 0.0)
                   {
                     f0_atoms(N , rcount ,lcount , f0 , ee);
	           }

                 if (strcmp(on,effec_f) == 0.0)
                   {
                     f0_mod_atoms(N , rcount ,lcount , f0 , ee_f0mod);
	           }
                   
             //    int ss , sss ;
             //    double final_sum_chk ;
            //     for (ss = 0; ss < rcount; ss++)
            //         {
            //             for (sss = 0; sss < lcount; sss++) 
            //                  {
            //                    final_sum_chk = final_sum_chk + f0[ss][sss] ;
            //                  }
            //         }

	//	  printf("rcount , lcount , f0 % d % d % le\n",rcount , lcount, final_sum_chk);
                  ////////////////////////////////////////////////////////////////////  
		  ////////////////////Input Q from Input File/////////////////////////  
		  ////////////////////////////////////////////////////////////////////

                  double *Q ;
		  Q = calloc (lcount, sizeof (double));
		  fscanf (Q_info, "%s\n", ee);

                 if (type_check == 0)
                    {
                     sprintf (tstep, "%08d/", ts);
                     strcat (strcat (ee, tstep), qdat);
                    }
		  if (type_check == 1)
                    {
                     sprintf (tstep, "%08d/", dummy_ts);
                     strcat (strcat (ee, tstep), qdat);
                    }
		 
                  Q_atoms(N ,lcount , Q , ee);

		  ////////////////////////////////////////////////////////////////////  
		  ///////////////////Input fpp from Input File////////////////////////  
		  ////////////////////////////////////////////////////////////////////
		  
                  
		  double (*fpp);
		  fpp = calloc (rcount, sizeof (double));	/* Exporting f'' Array */
                
		  fscanf (fpp_info, "%s\n", ee);
                
                  if (type_check == 0)
                    {
                     sprintf (tstep, "%08d/", ts);
                     strcat (strcat (ee, tstep), fppdat);
                    }
		  if (type_check == 1)
                    {
                     sprintf (tstep, "%08d/", dummy_ts);
                     strcat (strcat (ee, tstep), fppdat);
                    }
                  
                  fpp_atoms(rcount , fpp , ee , fpp_check);
                
                  ////////////////////////////////////////////////////////////////////  
		  ///////////Input Electron Coordinates from Input File///////////////  
		  ////////////////////////////////////////////////////////////////////
                  

		  double (*r_ele)[3];
		  r_ele = calloc (r_ele_variable * 3, sizeof (double));	
		//  printf("r_ele_variable % d\n",r_ele_variable);
                
		  fscanf (ele_r_info, "%s\n", ee);
                
                  if (type_check == 0)
                    {
                     sprintf (tstep, "%08d/", ts);
                     strcat (strcat (ee, tstep), ele_r);
                    }
		  if (type_check == 1)
                    {
                     sprintf (tstep, "%08d/", dummy_ts);
                     strcat (strcat (ee, tstep), ele_r);
                    }
                
                   electron_coordinates(N , r_ele_variable , r_ele , ee ) ;


                  ////////////////////////////////////////////////////////////////////  
		  //////////////////Input Charges from Input File/////////////////////  
		  ////////////////////////////////////////////////////////////////////
                  

		  int *q_ele;
		  q_ele = calloc (r_ele_variable , sizeof (int));	
                
		  fscanf (charge_info, "%s\n", ee);
                
                  if (type_check == 0)
                    {
                     sprintf (tstep, "%08d/", ts);
                     strcat (strcat (ee, tstep), q_charge);
                    }
		  if (type_check == 1)
                    {
                     sprintf (tstep, "%08d/", dummy_ts);
                     strcat (strcat (ee, tstep), q_charge);
                    }
                

                  charge_on_electrons(N , r_ele_variable , q_ele , ee) ;

		  ////////////////////////////////////////////////////////////////////
		  ////////////////////////////////////////////////////////////////////

                  ////////////////////////////////////////////////////////////////////  
		  /////////////////////Input Z no from Input File/////////////////////  
		  ////////////////////////////////////////////////////////////////////
                  

		  int *Z_atoms;
		  Z_atoms = calloc (N , sizeof (int));	
                
		  fscanf (atomic_no_info, "%s\n", ee);
                
                if (type_check == 0)
                    {
                     sprintf (tstep, "%08d/", ts);
                     strcat (strcat (ee, tstep), Zdat);
                    }
		  if (type_check == 1)
                    {
                     sprintf (tstep, "%08d/", dummy_ts);
                     strcat (strcat (ee, tstep), Zdat);
                    }
                
                  atomic_num_atoms(N  ,Z_atoms, ee) ;

                       
		  ////////////////////////////////////////////////////////////////////
		  ////////////////////////////////////////////////////////////////////

		  /* list of the file* ccq[l][0,1,2] , Q[r] , f0[r][rr] , &atyp[r] */
		  /*rvect1 * cq[i][0] + rvect2 * cq[i][1] + rvect3 * cq[i][2] */

		  for (l = 0; l < N; l++)
		    {
		      ccq[l][0] = cq[l][0]  + CELL[i][0] * scellx;
		      ccq[l][1] = cq[l][1]  + CELL[i][1] * scelly;
		      ccq[l][2] = cq[l][2]  + CELL[i][2] * scellz;
		      fprintf (coordinates_out, "%le %le %le\n", ccq[l][0],
			       ccq[l][1], ccq[l][2]);
		    }
                  ///////////////////////////////////////////////////////////////////
                  //////////////////////Ewald Sphere Calc////////////////////////////
                  ///////////////////////////////////////////////////////////////////
                  
                  int  ii=0, jj=0 , iii = 0;
                  double qvect1=0, qvect2=0, qvect3=0 , real_part=0, imagin_part=0 ,flu_part=0 ;
                  double  ri=0 , rj=0, rk=0, tt=0 , xx=0 , yy=0 , ewald_sphere=0 , mod_rvect=0;
                  double complex exp_part , exp_intensity_part ,f0_part;
                 
                  ewald_sphere = (2 * 2 * M_PI) / lambda ;
                  qc = qc + 1 ; 		  
               
                  if( qc < 2 )
                   {
                    strcpy(Q_log,"./output/q_points") ;
                    qpoints_out = fopen (Q_log, "w") ;
                   }
	            

                     for ( l = 0 ; l < qp_rcount ; l++ ) 
                        {
			 ri = qpt_array[l][0] ;
                         rj = qpt_array[l][1] ;
                         rk = qpt_array[l][2] ;

                         qvect1 = (2.0 * M_PI * ri) / ucellx;
                         qvect2 = (2.0 * M_PI * rj) / ucelly;
                         qvect3 = (2.0 * M_PI * rk) / ucellz;
                        
                         mod_rvect = sqrt (qvect1 * qvect1 +
                                           qvect2 * qvect2 +
                                           qvect3 * qvect3 ) ;
                          
	                 xx = 0 ; yy = 5e9 ; jj = 0;
	                 
                         for ( iii = 0; iii < lcount; iii++ )
	                     {
	                      xx = fabs (mod_rvect - (Q[iii] / 0.529177208));//converting q from a.u to Angstroms
	                      if (xx < yy)
	                       {
	                        yy = xx;
	                        jj = iii;
	                       }	// if loop ended
	                     }
                        
                         if( qc < 2 )
	                     {
	                      fprintf (qpoints_out, "% le  % le  % le\n", ri, rj, rk);
	                     }
                         
                         for ( ii = 0 ; ii< num_threads ; ii++ )
			     {
			       dummy_array_1[ii] = (double complex) 0.0;
			       dummy_array_2[ii] = (double complex) 0.0;
			       dummy_array_3[ii] = (double complex) 0.0;
			       dummy_array_4[ii] = (double complex) 0.0;
			       dummy_array_5[ii] = (double complex) 0.0;
			       dummy_array_6[ii] = (double complex) 0.0;
			       dummy_array_7[ii] = (double complex) 0.0;
			     }
                  


                        calc_intensity (qp_rcount ,i, N, rcount, lcount, jj, flu_array, total_sc, ts, T0, tau, integrated_signal, f0, atyp, fpp, 
                                        fpp_check, qvect1, qvect2, qvect3, cq, CELL, scellx,  scelly, scellz, Z_atoms , r_ele_variable, q_ele, r_ele, f_0,
                                        dummy_array_1, dummy_array_2, dummy_array_3, dummy_array_4, dummy_array_5, dummy_array_6, dummy_array_7 , qrterm_comp ,
                                        &comp_num, num_threads, chk_xxx, intensity_account, qrterm_account, copy_var_count, tpshapef, tpshapeg, st_snp, FinalSnap, 
                                        sp_diff, 
				      Hydrogen_form_factor_account,
				      Helium_form_factor_account,
				      Lithium_form_factor_account,
				      Beryllium_form_factor_account,
				      Boron_form_factor_account,
				      Carbon_form_factor_account,
				      Nitrogen_form_factor_account,
				      Oxygen_form_factor_account,
				      Fluorine_form_factor_account,
				      Neon_form_factor_account,
				      Sodium_form_factor_account,
				      Magnesium_form_factor_account,
				      Aluminum_form_factor_account,
				      Silicon_form_factor_account,
				      Phosphorus_form_factor_account,
				      Sulfur_form_factor_account,
				      Chlorine_form_factor_account,
				      Argon_form_factor_account,
				      Potassium_form_factor_account,
				      Calcium_form_factor_account,
				      Scandium_form_factor_account,
				      Titanium_form_factor_account,
				      Vanadium_form_factor_account,
				      Chromium_form_factor_account,
				      Manganese_form_factor_account,
				      Iron_form_factor_account,
				      Cobalt_form_factor_account,
				      Nickel_form_factor_account,
				      Copper_form_factor_account,
				      Zinc_form_factor_account,
				      Gallium_form_factor_account,
				      Germanium_form_factor_account,
				      Arsenic_form_factor_account,
				      Selenium_form_factor_account,
				      Bromine_form_factor_account,
				      Krypton_form_factor_account,
				      Rubidium_form_factor_account,
				      Strontium_form_factor_account,
				      Yttrium_form_factor_account,
				      Zirconium_form_factor_account,
				      Niobium_form_factor_account,
				      Molybdenum_form_factor_account,
				      Technetium_form_factor_account,
				      Ruthenium_form_factor_account,
				      Rhodium_form_factor_account,
				      Palladium_form_factor_account,
				      Silver_form_factor_account,
				      Cadmium_form_factor_account,
				      Indium_form_factor_account,
				      Tin_form_factor_account,
				      Antimony_form_factor_account,
				      Tellurium_form_factor_account,
				      Iodine_form_factor_account,
				      Xenon_form_factor_account,
				      Cesium_form_factor_account,
				      Barium_form_factor_account,
				      Lanthanum_form_factor_account,
				      Cerium_form_factor_account,
				      Praseodymium_form_factor_account,
				      Neodymium_form_factor_account,
				      Promethium_form_factor_account,
				      Samarium_form_factor_account,
				      Europium_form_factor_account,
				      Gadolinium_form_factor_account,
				      Terbium_form_factor_account,
				      Dysprosium_form_factor_account,
				      Holmium_form_factor_account,
				      Erbium_form_factor_account,
				      Thulium_form_factor_account,
				      Ytterbium_form_factor_account,
				      Lutetium_form_factor_account,
				      Hafnium_form_factor_account,
				      Tantalum_form_factor_account,
				      Tungsten_form_factor_account,
				      Rhenium_form_factor_account,
				      Osmium_form_factor_account,
				      Iridium_form_factor_account,
				      Platinum_form_factor_account,
				      Gold_form_factor_account,
				      Mercury_form_factor_account,
				      Thallium_form_factor_account,
				      Lead_form_factor_account,
				      Bismuth_form_factor_account,
				      Polonium_form_factor_account,
				      Astatine_form_factor_account,
				      Radon_form_factor_account,
				      Francium_form_factor_account,
				      Radium_form_factor_account,
				      Actinium_form_factor_account,
				      Thorium_form_factor_account,
				      Protactinium_form_factor_account,
				      Uranium_form_factor_account,
				      Neptunium_form_factor_account,
				      Plutonium_form_factor_account,
				      Americium_form_factor_account,
				      Curium_form_factor_account,
				      Berkelium_form_factor_account,
				      Californium_form_factor_account,
				      Einsteinium_form_factor_account,
				      Fermium_form_factor_account,
				      Mendelevium_form_factor_account,
				      Nobelium_form_factor_account,
				      Lawrencium_form_factor_account,
				      Rutherfordium_form_factor_account,
				      Dubnium_form_factor_account,
				      Seaborgium_form_factor_account,
				      Bohrium_form_factor_account,
				      Hassium_form_factor_account,
				      Meitnerium_form_factor_account,
									      
				      Hydrogen_form_factor,
				      Helium_form_factor,
				      Lithium_form_factor,
				      Beryllium_form_factor,
				      Boron_form_factor,
				      Carbon_form_factor,
				      Nitrogen_form_factor,
				      Oxygen_form_factor,
				      Fluorine_form_factor,
				      Neon_form_factor,
				      Sodium_form_factor,
				      Magnesium_form_factor,
				      Aluminum_form_factor,
				      Silicon_form_factor,
				      Phosphorus_form_factor,
				      Sulfur_form_factor,
				      Chlorine_form_factor,
				      Argon_form_factor,
				      Potassium_form_factor,
				      Calcium_form_factor,
				      Scandium_form_factor,
				      Titanium_form_factor,
				      Vanadium_form_factor,
				      Chromium_form_factor,
				      Manganese_form_factor,
				      Iron_form_factor,
				      Cobalt_form_factor,
				      Nickel_form_factor,
				      Copper_form_factor,
				      Zinc_form_factor,
				      Gallium_form_factor,
				      Germanium_form_factor,
				      Arsenic_form_factor,
				      Selenium_form_factor,
				      Bromine_form_factor,
				      Krypton_form_factor,
				      Rubidium_form_factor,
				      Strontium_form_factor,
				      Yttrium_form_factor,
				      Zirconium_form_factor,
				      Niobium_form_factor,
				      Molybdenum_form_factor,
				      Technetium_form_factor,
				      Ruthenium_form_factor,
				      Rhodium_form_factor,
				      Palladium_form_factor,
				      Silver_form_factor,
				      Cadmium_form_factor,
				      Indium_form_factor,
				      Tin_form_factor,
				      Antimony_form_factor,
				      Tellurium_form_factor,
				      Iodine_form_factor,
				      Xenon_form_factor,
				      Cesium_form_factor,
				      Barium_form_factor,
				      Lanthanum_form_factor,
				      Cerium_form_factor,
				      Praseodymium_form_factor,
				      Neodymium_form_factor,
				      Promethium_form_factor,
				      Samarium_form_factor,
				      Europium_form_factor,
				      Gadolinium_form_factor,
				      Terbium_form_factor,
				      Dysprosium_form_factor,
				      Holmium_form_factor,
				      Erbium_form_factor,
				      Thulium_form_factor,
				      Ytterbium_form_factor,
				      Lutetium_form_factor,
				      Hafnium_form_factor,
				      Tantalum_form_factor,
				      Tungsten_form_factor,
				      Rhenium_form_factor,
				      Osmium_form_factor,
				      Iridium_form_factor,
				      Platinum_form_factor,
				      Gold_form_factor,
				      Mercury_form_factor,
				      Thallium_form_factor,
				      Lead_form_factor,
				      Bismuth_form_factor,
				      Polonium_form_factor,
				      Astatine_form_factor,
				      Radon_form_factor,
				      Francium_form_factor,
				      Radium_form_factor,
				      Actinium_form_factor,
				      Thorium_form_factor,
				      Protactinium_form_factor,
				      Uranium_form_factor,
				      Neptunium_form_factor,
				      Plutonium_form_factor,
				      Americium_form_factor,
				      Curium_form_factor,
				      Berkelium_form_factor,
				      Californium_form_factor,
				      Einsteinium_form_factor,
				      Fermium_form_factor,
				      Mendelevium_form_factor,
				      Nobelium_form_factor,
				      Lawrencium_form_factor,
				      Rutherfordium_form_factor,
				      Dubnium_form_factor,
				      Seaborgium_form_factor,
				      Bohrium_form_factor,
				      Hassium_form_factor,
				      Meitnerium_form_factor) ;

        #ifdef USE_CUDA 
        cuda_function(i, N, total_sc, flu_part, lcount, rcount, f0, atyp, jj, qvect1, qvect2, qvect3, cq, CELL, scellx, scelly,  scellz,dummy_array_1, dummy_array_2 );
        #endif
}	
                     if(qc < 2){fclose (qpoints_out);}
		  free (cq);
                  free (Q);
		  free (f0);
                  free (fpp);
                  free (r_ele);
                  free (q_ele);
                  free (Z_atoms);
                  free (atyp);
                  total_sc = total_sc + 1 ;
                  copy_var_count++ ;
		 
                 
                     //////////////////////////////////////////////////////////////////
		     //////////////////////////////////////////////////////////////////
		     //////////////////////////////////////////////////////////////////
                     
                    }
        /*************************************************************************************************/
        /*************************************************************************************************/
        /*************************************************************************************************/
                 else
                    {

                    for( xxxx = 0 ; xxxx < copy_var_count ; xxxx++ )
                     {
                      if (strcmp(chk_string[xxxx],copy_var) == 0)
                       {
                         str_num = xxxx ;
                         verify_num++ ;
                       }                          
                      }
                    }
                      
        /*************************************************************************************************/
        /*************************************************************************************************/
        /*************************************************************************************************/
                 if ( 0 == verify_num && 0 != i * spi_switch )   
                    {
                      strcpy(chk_string[copy_var_count] , copy_var ) ;
		     
		  ////////////////////////////////////////////////////////////////////  
		  /////////////////Input Coordinates from Input File//////////////////  
		  ////////////////////////////////////////////////////////////////////
                 
                  double (*cq)[3];
                  cq = calloc (N * 3, sizeof (double));	/* Exporting Coordinates Array */
                  coordinates_atoms( N , cq , ee) ;                  


		  ////////////////////////////////////////////////////////////////////  
		  ////////////////Input Species Type from Input File//////////////////  
		  ////////////////////////////////////////////////////////////////////
                  int *atyp ;
                  atyp = calloc (N, sizeof (int));	/* Exporting Charges Array */
		  
                  
                   if (strcmp(on,effec_f) == 0.0)
                      {
                        strcpy(ee_atypmod,"./atyp_mod.dat");
                        species_atoms_mod( N , atyp , ee_atypmod);
                  //     printf ("modified atom type path % s\n",ee_atypmod);
                      }
                  
                   if (strcmp(off,effec_f) == 0.0)
                      {
                        fscanf (T_info, "%s\n", ee);

		        if (type_check == 0)
                           {
                            sprintf (tstep, "%08d/", ts);
                            strcat (strcat (ee, tstep), tdat);
                           }
		        if (type_check == 1)
                           {
                            sprintf (tstep, "%08d/", dummy_ts);
                            strcat (strcat (ee, tstep), tdat);
                           }
                        species_atoms( N , atyp , ee) ;
                      }
		  ////////////////////////////////////////////////////////////////////  
		  //////////////Scanning f0 & f0_mod from the Input File//////////////  
		  ////////////////////////////////////////////////////////////////////     


		  int ch;
		  int rcount = 0;
		  int lcount = 0;
                 
                 if (strcmp(on,effec_f) == 0.0)
                    {
                   //  fscanf (f0_modified, "%s\n", ee);
                       strcpy(ee_f0mod,"./f0_mod.dat");
		       FILE *fp = fopen (ee_f0mod, "r");
		       do
		        {
		         ch = fgetc (fp);
		         if (ch == '\n')
			 rcount++;
		        }
		       while (ch != EOF);
		       fclose (fp);


		       fp = fopen (ee_f0mod, "r");
		       while (0 == fscanf (fp, "%*s"))
		       {
		         lcount++;
		       }
		       fclose (fp);

		       lcount = lcount / rcount;
                    }
                    
                 if (strcmp(off,effec_f) == 0.0)
                    {
		          fscanf (f0_info, "%s\n", ee);
            
                          if (type_check == 0)
                              {
                               sprintf (tstep, "%08d/", ts);
                               strcat (strcat (ee, tstep), f0dat);
                               }
		          if (type_check == 1)
                               {
                               sprintf (tstep, "%08d/", dummy_ts);
                               strcat (strcat (ee, tstep), f0dat);
                               }
              		  FILE *fp = fopen (ee, "r");
	         	  do
		           {
		            ch = fgetc (fp);
		            if (ch == '\n')
			    rcount++;
		           }
		          while (ch != EOF);
		          fclose (fp);


		          fp = fopen (ee, "r");
		          while (0 == fscanf (fp, "%*s"))
		          {
		           lcount++;
		          }
		          fclose (fp);

		          lcount = lcount / rcount;
                    }

		  double (*f0)[lcount];
		  f0 = calloc (rcount * lcount, sizeof (double));	/* Exporting Charges Array */
		  ////////////////////////////////////////////////////////////////////  
		  /////////////////Input f0 & fo_mod from Input File//////////////////  
		  ////////////////////////////////////////////////////////////////////
                  

                 if (strcmp(off,effec_f) == 0.0)
                   {
                     f0_atoms(N , rcount ,lcount , f0 , ee);
	           }

                 if (strcmp(on,effec_f) == 0.0)
                   {
                     f0_mod_atoms(N , rcount ,lcount , f0 , ee_f0mod);
	           }
                  ////////////////////////////////////////////////////////////////////  
		  ////////////////////Input Q from Input File/////////////////////////  
		  ////////////////////////////////////////////////////////////////////

                  double *Q ;
		  Q = calloc (lcount, sizeof (double));
		  fscanf (Q_info, "%s\n", ee);
                 
                  if (type_check == 0)
                    {
                     sprintf (tstep, "%08d/", ts);
                     strcat (strcat (ee, tstep), qdat);
                    }
		  if (type_check == 1)
                    {
                     sprintf (tstep, "%08d/", dummy_ts);
                     strcat (strcat (ee, tstep), qdat);
                    }

                     Q_atoms(N ,lcount , Q , ee); 

		  ////////////////////////////////////////////////////////////////////  
		  ///////////////////Input fpp from Input File////////////////////////  
		  ////////////////////////////////////////////////////////////////////
		  
                  
		  double (*fpp);
		  fpp = calloc (rcount, sizeof (double));	/* Exporting f'' Array */
                
		  fscanf (fpp_info, "%s\n", ee);
                
                if (type_check == 0)
                    {
                     sprintf (tstep, "%08d/", ts);
                     strcat (strcat (ee, tstep), fppdat);
                    }
		  if (type_check == 1)
                    {
                     sprintf (tstep, "%08d/", dummy_ts);
                     strcat (strcat (ee, tstep), fppdat);
                    }
                

                  fpp_atoms(rcount , fpp , ee , fpp_check);
                
                  ////////////////////////////////////////////////////////////////////  
		  ///////////Input Electron Coordinates from Input File///////////////  
		  ////////////////////////////////////////////////////////////////////
                  

		  double (*r_ele)[3];
		  r_ele = calloc (r_ele_variable * 3, sizeof (double));	
		  fscanf (ele_r_info, "%s\n", ee);
                
                if (type_check == 0)
                    {
                     sprintf (tstep, "%08d/", ts);
                     strcat (strcat (ee, tstep), ele_r);
                    }
		  if (type_check == 1)
                    {
                     sprintf (tstep, "%08d/", dummy_ts);
                     strcat (strcat (ee, tstep), ele_r);
                    }
               
                 electron_coordinates(N , r_ele_variable , r_ele , ee ) ;

                  ////////////////////////////////////////////////////////////////////  
		  //////////////////Input Charges from Input File/////////////////////  
		  ////////////////////////////////////////////////////////////////////
                  

		  int *q_ele;
		  q_ele = calloc (r_ele_variable , sizeof (int));	
                
		  fscanf (charge_info, "%s\n", ee);
                
                if (type_check == 0)
                    {
                     sprintf (tstep, "%08d/", ts);
                     strcat (strcat (ee, tstep), q_charge);
                    }
		  if (type_check == 1)
                    {
                     sprintf (tstep, "%08d/", dummy_ts);
                     strcat (strcat (ee, tstep), q_charge);
                    }
                
                  charge_on_electrons(N , r_ele_variable , q_ele , ee) ;
		 
		  ////////////////////////////////////////////////////////////////////
		  ////////////////////////////////////////////////////////////////////

                  ////////////////////////////////////////////////////////////////////  
		  /////////////////////Input Z no from Input File/////////////////////  
		  ////////////////////////////////////////////////////////////////////
                  

		  int *Z_atoms;
		  Z_atoms = calloc (N , sizeof (int));	
                
		  fscanf (atomic_no_info, "%s\n", ee);
                
                if (type_check == 0)
                    {
                     sprintf (tstep, "%08d/", ts);
                     strcat (strcat (ee, tstep), Zdat);
                    }
		  if (type_check == 1)
                    {
                     sprintf (tstep, "%08d/", dummy_ts);
                     strcat (strcat (ee, tstep), Zdat);
                    }
                
                  atomic_num_atoms(N  ,Z_atoms, ee) ;


		  ////////////////////////////////////////////////////////////////////
		  ////////////////////////////////////////////////////////////////////

		  /* list of the file* ccq[l][0,1,2] , Q[r] , f0[r][rr] , &atyp[r] */
		  /*rvect1 * cq[i][0] + rvect2 * cq[i][1] + rvect3 * cq[i][2] */

		  for (l = 0; l < N; l++)
		    {
		      ccq[l][0] = cq[l][0] + CELL[i][0] * scellx;
		      ccq[l][1] = cq[l][1] + CELL[i][1] * scelly;
		      ccq[l][2] = cq[l][2] + CELL[i][2] * scellz;
		      fprintf (coordinates_out, "%le %le %le\n", ccq[l][0],
			       ccq[l][1], ccq[l][2]);
		    }
                  ///////////////////////////////////////////////////////////////////
                  //////////////////////Ewald Sphere Calc////////////////////////////
                  ///////////////////////////////////////////////////////////////////

               //   printf(" We are in the Second part Now\n " );
                  int  ii=0, jj=0 , iii = 0 ;
                  double qvect1=0, qvect2=0, qvect3=0 , real_part=0, imagin_part=0 ;
                  double ri=0 , rj=0, rk=0, tt=0 , xx=0 , yy=0 , ewald_sphere=0 , mod_rvect=0 , flu_part=0 ;
                  double complex exp_part , exp_intensity_part , f0_part;
                 
                  ewald_sphere = (2 * 2 * M_PI) / lambda ;
                  qc = qc + 1 ; 		  
               
                  if( qc < 2 )
                   {
                    strcpy(Q_log,"./output/q_points") ;
                    qpoints_out = fopen (Q_log, "w") ;
                   }
	          
                  for ( l = 0 ; l < qp_rcount ; l++ ) 
                      {
		        ri = qpt_array[l][0] ;
                        rj = qpt_array[l][1] ;
                        rk = qpt_array[l][2] ;

                        qvect1 = (2.0 * M_PI * ri) / ucellx;
                        qvect2 = (2.0 * M_PI * rj) / ucelly;
                        qvect3 = (2.0 * M_PI * rk) / ucellz;
                 
                        mod_rvect = sqrt (qvect1 * qvect1 +
                                          qvect2 * qvect2 + 
                                          qvect3 * qvect3 ) ;
                          
	                xx = 0 ; yy = 5e9 ; jj = 0;
	            
                    for ( iii = 0; iii < lcount; iii++ )
	                {
	                  xx = fabs (mod_rvect - (Q[iii] / 0.529177208));//converting q from a.u to Angstroms
	           
                           if (xx < yy)
	                     {
	                        yy = xx;
	                        jj = iii;
	                     }	// if loop ended
	                }
                        
                        if(qc < 2)
	                     {
	                      fprintf (qpoints_out, "% le  % le  % le\n", ri, rj, rk);
	                     }
                           for (ii = 0 ; ii< num_threads ; ii++)
			    {
			     dummy_array_1[ii] = (double complex) 0.0;
			     dummy_array_2[ii] = (double complex) 0.0;
			     dummy_array_3[ii] = (double complex) 0.0;
			     dummy_array_4[ii] = (double complex) 0.0;
                             dummy_array_5[ii] = (double complex) 0.0;
                             dummy_array_6[ii] = (double complex) 0.0;
                             dummy_array_7[ii] = (double complex) 0.0;
			    }
			  
                          


                          
                   //       flu_part = sqrt(flu_array[total_sc]) *sqrt(exp(-((ts-T0)*(ts-T0))/(tau*tau))) ;
       #ifdef USE_CUDA
       cuda_function(N, total_sc, flu_part, lcount, rcount, f0, atyp, jj, qvect1, qvect2, qvect3, cq, CELL, scellx, scelly, scellz, dummy_array_1, dummy_array_2 );
       #endif

        calc_intensity (qp_rcount, i, N, rcount, lcount, jj, flu_array, total_sc, ts, T0, tau, integrated_signal, f0, atyp, fpp, 
                        fpp_check, qvect1, qvect2, qvect3, cq, CELL, scellx,  scelly, scellz,Z_atoms , r_ele_variable, q_ele, r_ele, f_0,
                        dummy_array_1, dummy_array_2, dummy_array_3, dummy_array_4, dummy_array_5, dummy_array_6, dummy_array_7, qrterm_comp, 
                        &comp_num, num_threads, chk_xxx, intensity_account, qrterm_account, copy_var_count, tpshapef, tpshapeg, st_snp, 
                        FinalSnap, sp_diff, 
			Hydrogen_form_factor_account,
			Helium_form_factor_account,
			Lithium_form_factor_account,
			Beryllium_form_factor_account,
			Boron_form_factor_account,
			Carbon_form_factor_account,
			Nitrogen_form_factor_account,
			Oxygen_form_factor_account,
			Fluorine_form_factor_account,
			Neon_form_factor_account,
			Sodium_form_factor_account,
			Magnesium_form_factor_account,
			Aluminum_form_factor_account,
			Silicon_form_factor_account,
			Phosphorus_form_factor_account,
			Sulfur_form_factor_account,
			Chlorine_form_factor_account,
			Argon_form_factor_account,
			Potassium_form_factor_account,
			Calcium_form_factor_account,
			Scandium_form_factor_account,
			Titanium_form_factor_account,
			Vanadium_form_factor_account,
			Chromium_form_factor_account,
			Manganese_form_factor_account,
			Iron_form_factor_account,
			Cobalt_form_factor_account,
			Nickel_form_factor_account,
			Copper_form_factor_account,
			Zinc_form_factor_account,
			Gallium_form_factor_account,
			Germanium_form_factor_account,
			Arsenic_form_factor_account,
			Selenium_form_factor_account,
			Bromine_form_factor_account,
			Krypton_form_factor_account,
			Rubidium_form_factor_account,
			Strontium_form_factor_account,
			Yttrium_form_factor_account,
			Zirconium_form_factor_account,
			Niobium_form_factor_account,
			Molybdenum_form_factor_account,
			Technetium_form_factor_account,
			Ruthenium_form_factor_account,
			Rhodium_form_factor_account,
			Palladium_form_factor_account,
			Silver_form_factor_account,
			Cadmium_form_factor_account,
			Indium_form_factor_account,
			Tin_form_factor_account,
			Antimony_form_factor_account,
			Tellurium_form_factor_account,
			Iodine_form_factor_account,
			Xenon_form_factor_account,
			Cesium_form_factor_account,
			Barium_form_factor_account,
			Lanthanum_form_factor_account,
			Cerium_form_factor_account,
			Praseodymium_form_factor_account,
			Neodymium_form_factor_account,
			Promethium_form_factor_account,
			Samarium_form_factor_account,
			Europium_form_factor_account,
			Gadolinium_form_factor_account,
			Terbium_form_factor_account,
			Dysprosium_form_factor_account,
			Holmium_form_factor_account,
			Erbium_form_factor_account,
			Thulium_form_factor_account,
			Ytterbium_form_factor_account,
			Lutetium_form_factor_account,
			Hafnium_form_factor_account,
			Tantalum_form_factor_account,
			Tungsten_form_factor_account,
			Rhenium_form_factor_account,
			Osmium_form_factor_account,
			Iridium_form_factor_account,
			Platinum_form_factor_account,
			Gold_form_factor_account,
			Mercury_form_factor_account,
			Thallium_form_factor_account,
			Lead_form_factor_account,
			Bismuth_form_factor_account,
			Polonium_form_factor_account,
			Astatine_form_factor_account,
			Radon_form_factor_account,
			Francium_form_factor_account,
			Radium_form_factor_account,
			Actinium_form_factor_account,
			Thorium_form_factor_account,
			Protactinium_form_factor_account,
			Uranium_form_factor_account,
			Neptunium_form_factor_account,
			Plutonium_form_factor_account,
			Americium_form_factor_account,
			Curium_form_factor_account,
			Berkelium_form_factor_account,
			Californium_form_factor_account,
			Einsteinium_form_factor_account,
			Fermium_form_factor_account,
			Mendelevium_form_factor_account,
			Nobelium_form_factor_account,
			Lawrencium_form_factor_account,
			Rutherfordium_form_factor_account,
			Dubnium_form_factor_account,
			Seaborgium_form_factor_account,
			Bohrium_form_factor_account,
			Hassium_form_factor_account,
			Meitnerium_form_factor_account,
								
			Hydrogen_form_factor,
			Helium_form_factor,
			Lithium_form_factor,
			Beryllium_form_factor,
			Boron_form_factor,
			Carbon_form_factor,
			Nitrogen_form_factor,
			Oxygen_form_factor,
			Fluorine_form_factor,
			Neon_form_factor,
			Sodium_form_factor,
			Magnesium_form_factor,
			Aluminum_form_factor,
			Silicon_form_factor,
			Phosphorus_form_factor,
			Sulfur_form_factor,
			Chlorine_form_factor,
			Argon_form_factor,
			Potassium_form_factor,
			Calcium_form_factor,
			Scandium_form_factor,
			Titanium_form_factor,
			Vanadium_form_factor,
			Chromium_form_factor,
			Manganese_form_factor,
			Iron_form_factor,
			Cobalt_form_factor,
			Nickel_form_factor,
			Copper_form_factor,
			Zinc_form_factor,
			Gallium_form_factor,
			Germanium_form_factor,
			Arsenic_form_factor,
			Selenium_form_factor,
			Bromine_form_factor,
			Krypton_form_factor,
			Rubidium_form_factor,
			Strontium_form_factor,
			Yttrium_form_factor,
			Zirconium_form_factor,
			Niobium_form_factor,
			Molybdenum_form_factor,
			Technetium_form_factor,
			Ruthenium_form_factor,
			Rhodium_form_factor,
			Palladium_form_factor,
			Silver_form_factor,
			Cadmium_form_factor,
			Indium_form_factor,
			Tin_form_factor,
			Antimony_form_factor,
			Tellurium_form_factor,
			Iodine_form_factor,
			Xenon_form_factor,
			Cesium_form_factor,
			Barium_form_factor,
			Lanthanum_form_factor,
			Cerium_form_factor,
			Praseodymium_form_factor,
			Neodymium_form_factor,
			Promethium_form_factor,
			Samarium_form_factor,
			Europium_form_factor,
			Gadolinium_form_factor,
			Terbium_form_factor,
			Dysprosium_form_factor,
			Holmium_form_factor,
			Erbium_form_factor,
			Thulium_form_factor,
			Ytterbium_form_factor,
			Lutetium_form_factor,
			Hafnium_form_factor,
			Tantalum_form_factor,
			Tungsten_form_factor,
			Rhenium_form_factor,
			Osmium_form_factor,
			Iridium_form_factor,
			Platinum_form_factor,
			Gold_form_factor,
			Mercury_form_factor,
			Thallium_form_factor,
			Lead_form_factor,
			Bismuth_form_factor,
			Polonium_form_factor,
			Astatine_form_factor,
			Radon_form_factor,
			Francium_form_factor,
			Radium_form_factor,
			Actinium_form_factor,
			Thorium_form_factor,
			Protactinium_form_factor,
			Uranium_form_factor,
			Neptunium_form_factor,
			Plutonium_form_factor,
			Americium_form_factor,
			Curium_form_factor,
			Berkelium_form_factor,
			Californium_form_factor,
			Einsteinium_form_factor,
			Fermium_form_factor,
			Mendelevium_form_factor,
			Nobelium_form_factor,
			Lawrencium_form_factor,
			Rutherfordium_form_factor,
			Dubnium_form_factor,
			Seaborgium_form_factor,
			Bohrium_form_factor,
			Hassium_form_factor,
			Meitnerium_form_factor) ;
  

			}
	
        /*************************************************************************************************/
        /*************************************************************************************************/
        /*************************************************************************************************/
                  if(qc < 2){fclose (qpoints_out);}
		  free (cq);
                  free (Q);
		  free (f0);
		  free (fpp);
                  free (r_ele);
                  free (q_ele);
                  free (Z_atoms);
                  free (atyp) ;
                  total_sc = total_sc + 1 ;
                  copy_var_count++ ;
                    }
                    
        /*************************************************************************************************/
        /*************************************************************************************************/
        /*************************************************************************************************/
         
                if (0 != spi_switch * verify_num )   
		     {
                       //  printf("string in the last%s\n", copy_var);
                    /*   double (*cq)[3];
                       cq = calloc (N * 3, sizeof (double));	
                       coordinates_atoms( N , cq , ee) ; 
		        for (l = 0; l < N; l++)
		         {
		           ccq[l][0] = cq[l][0]  + CELL[i][0] * scellx;
		           ccq[l][1] = cq[l][1]  + CELL[i][1] * scelly;
		           ccq[l][2] = cq[l][2]  + CELL[i][2] * scellz;
		           fprintf (coordinates_out, "%le %le %le\n", ccq[l][0],
		                                	  ccq[l][1], ccq[l][2]);
        	         }*/
                       fscanf(T_info, "%s\n", ee) ;
                       fscanf(Q_info, "%s\n", ee) ;
                       fscanf(f0_info, "%s\n", ee) ;
                       fscanf(fpp_info, "%s\n", ee) ;
                       fscanf(ele_r_info, "%s\n", ee) ;
                       fscanf(charge_info, "%s\n", ee) ;
                       fscanf(atomic_no_info, "%s\n", ee) ;
                      
                       double ri=0 , rj=0, rk=0 , qvect1=0, qvect2=0, qvect3=0;
                       double  ewald_sphere=0 , mod_rvect=0;
		       ewald_sphere = (2 * 2 * M_PI) / lambda ;
		       double complex new_g = 0 ;
      //                 struct timespec tht1,tht2;
    //                   clock_gettime(CLOCK_REALTIME, &tht1);
                       #pragma omp parallel private(ri,rj,rk,qvect1,qvect2,qvect3) 
                         {
                          my_id = omp_get_thread_num() ;
		          for ( l = my_id ; l < qp_rcount ; l += num_threads) 
                              {
			       ri = qpt_array[l][0] ;
                               rj = qpt_array[l][1] ;
                               rk = qpt_array[l][2] ;

                               qvect1 = (2.0 * M_PI * ri) / ucellx;
                               qvect2 = (2.0 * M_PI * rj) / ucelly;
                               qvect3 = (2.0 * M_PI * rk) / ucellz;
                            //   f_0[l] += (intensity_account[l][str_num] *
		              //           cexp(  (( qvect1 * CELL[i][0]* scellx) +
		                //                     ( qvect2 * CELL[i][1]* scelly) +
			          //                   ( qvect3 * CELL[i][2]* scellz)) * I ) ) ;
                               f_0[l] += (intensity_account[str_num][l] *
		                         cexp(  (( qvect1 * CELL[i][0]* scellx) +
		                                     ( qvect2 * CELL[i][1]* scelly) +
			                             ( qvect3 * CELL[i][2]* scellz)) * I ) ) ;
                             
                        //       qrterm_comp[l] += (qrterm_account[l][str_num] *
		          //               cexp(  (( qvect1 * CELL[i][0]* scellx) +
		            //                         ( qvect2 * CELL[i][1]* scelly) +
			      //                       ( qvect3 * CELL[i][2]* scellz)) * I ) ) ;
                               qrterm_comp[l] += (qrterm_account[str_num][l] *
		                         cexp(  (( qvect1 * CELL[i][0]* scellx) +
		                                     ( qvect2 * CELL[i][1]* scelly) +
			                             ( qvect3 * CELL[i][2]* scellz)) * I ) ) ;
                                 
			      Hydrogen_form_factor[l] +=	Hydrogen_form_factor_account[str_num][l];
			      Helium_form_factor[l] +=		Helium_form_factor_account[str_num][l];
			      Lithium_form_factor[l] +=		Lithium_form_factor_account[str_num][l];
			      Beryllium_form_factor[l] +=	Beryllium_form_factor_account[str_num][l];
			      Boron_form_factor[l] +=		Boron_form_factor_account[str_num][l];
			      Carbon_form_factor[l] +=		Carbon_form_factor_account[str_num][l];
			      Nitrogen_form_factor[l] +=	Nitrogen_form_factor_account[str_num][l];
			      Oxygen_form_factor[l] +=		Oxygen_form_factor_account[str_num][l];
			      Fluorine_form_factor[l] +=	Fluorine_form_factor_account[str_num][l];
			      Neon_form_factor[l] +=		Neon_form_factor_account[str_num][l];
			      Sodium_form_factor[l] +=		Sodium_form_factor_account[str_num][l];
			      Magnesium_form_factor[l] +=	Magnesium_form_factor_account[str_num][l];
			      Aluminum_form_factor[l] +=	Aluminum_form_factor_account[str_num][l];
			      Silicon_form_factor[l] +=		Silicon_form_factor_account[str_num][l];
			      Phosphorus_form_factor[l] +=	Phosphorus_form_factor_account[str_num][l];
			      Sulfur_form_factor[l] +=		Sulfur_form_factor_account[str_num][l];
			      Chlorine_form_factor[l] +=	Chlorine_form_factor_account[str_num][l];
			      Argon_form_factor[l] +=		Argon_form_factor_account[str_num][l];
			      Potassium_form_factor[l] +=	Potassium_form_factor_account[str_num][l];
			      Calcium_form_factor[l] +=		Calcium_form_factor_account[str_num][l];
			      Scandium_form_factor[l] +=	Scandium_form_factor_account[str_num][l];
			      Titanium_form_factor[l] +=	Titanium_form_factor_account[str_num][l];
			      Vanadium_form_factor[l] +=	Vanadium_form_factor_account[str_num][l];
			      Chromium_form_factor[l] +=	Chromium_form_factor_account[str_num][l];
			      Manganese_form_factor[l] +=	Manganese_form_factor_account[str_num][l];
			      Iron_form_factor[l] +=		Iron_form_factor_account[str_num][l];
			      Cobalt_form_factor[l] +=		Cobalt_form_factor_account[str_num][l];
			      Nickel_form_factor[l] +=		Nickel_form_factor_account[str_num][l];
			      Copper_form_factor[l] +=		Copper_form_factor_account[str_num][l];
			      Zinc_form_factor[l] +=		Zinc_form_factor_account[str_num][l];
			      Gallium_form_factor[l] +=		Gallium_form_factor_account[str_num][l];
			      Germanium_form_factor[l] +=	Germanium_form_factor_account[str_num][l];
			      Arsenic_form_factor[l] +=		Arsenic_form_factor_account[str_num][l];
			      Selenium_form_factor[l] +=	Selenium_form_factor_account[str_num][l];
			      Bromine_form_factor[l] +=		Bromine_form_factor_account[str_num][l];
			      Krypton_form_factor[l] +=		Krypton_form_factor_account[str_num][l];
			      Rubidium_form_factor[l] +=	Rubidium_form_factor_account[str_num][l];
			      Strontium_form_factor[l] +=	Strontium_form_factor_account[str_num][l];
			      Yttrium_form_factor[l] +=		Yttrium_form_factor_account[str_num][l];
			      Zirconium_form_factor[l] +=	Zirconium_form_factor_account[str_num][l];
			      Niobium_form_factor[l] +=		Niobium_form_factor_account[str_num][l];
			      Molybdenum_form_factor[l] +=	Molybdenum_form_factor_account[str_num][l];
			      Technetium_form_factor[l] +=	Technetium_form_factor_account[str_num][l];
			      Ruthenium_form_factor[l] +=	Ruthenium_form_factor_account[str_num][l];
			      Rhodium_form_factor[l] +=		Rhodium_form_factor_account[str_num][l];
			      Palladium_form_factor[l] +=	Palladium_form_factor_account[str_num][l];
			      Silver_form_factor[l] +=		Silver_form_factor_account[str_num][l];
			      Cadmium_form_factor[l] +=		Cadmium_form_factor_account[str_num][l];
			      Indium_form_factor[l] +=		Indium_form_factor_account[str_num][l];
			      Tin_form_factor[l] +=		Tin_form_factor_account[str_num][l];
			      Antimony_form_factor[l] +=	Antimony_form_factor_account[str_num][l];
			      Tellurium_form_factor[l] +=	Tellurium_form_factor_account[str_num][l];
			      Iodine_form_factor[l] +=		Iodine_form_factor_account[str_num][l];
			      Xenon_form_factor[l] +=		Xenon_form_factor_account[str_num][l];
			      Cesium_form_factor[l] +=		Cesium_form_factor_account[str_num][l];
			      Barium_form_factor[l] +=		Barium_form_factor_account[str_num][l];
			      Lanthanum_form_factor[l] +=	Lanthanum_form_factor_account[str_num][l];
			      Cerium_form_factor[l] +=		Cerium_form_factor_account[str_num][l];
			      Praseodymium_form_factor[l] +=	Praseodymium_form_factor_account[str_num][l];
			      Neodymium_form_factor[l] +=	Neodymium_form_factor_account[str_num][l];
			      Promethium_form_factor[l] +=	Promethium_form_factor_account[str_num][l];
			      Samarium_form_factor[l] +=	Samarium_form_factor_account[str_num][l];
			      Europium_form_factor[l] +=	Europium_form_factor_account[str_num][l];
			      Gadolinium_form_factor[l] +=	Gadolinium_form_factor_account[str_num][l];
			      Terbium_form_factor[l] +=		Terbium_form_factor_account[str_num][l];
			      Dysprosium_form_factor[l] +=	Dysprosium_form_factor_account[str_num][l];
			      Holmium_form_factor[l] +=		Holmium_form_factor_account[str_num][l];
			      Erbium_form_factor[l] +=		Erbium_form_factor_account[str_num][l];
			      Thulium_form_factor[l] +=		Thulium_form_factor_account[str_num][l];
			      Ytterbium_form_factor[l] +=	Ytterbium_form_factor_account[str_num][l];
			      Lutetium_form_factor[l] +=	Lutetium_form_factor_account[str_num][l];
			      Hafnium_form_factor[l] +=		Hafnium_form_factor_account[str_num][l];
			      Tantalum_form_factor[l] +=	Tantalum_form_factor_account[str_num][l];
			      Tungsten_form_factor[l] +=	Tungsten_form_factor_account[str_num][l];
			      Rhenium_form_factor[l] +=		Rhenium_form_factor_account[str_num][l];
			      Osmium_form_factor[l] +=		Osmium_form_factor_account[str_num][l];
			      Iridium_form_factor[l] +=		Iridium_form_factor_account[str_num][l];
			      Platinum_form_factor[l] +=	Platinum_form_factor_account[str_num][l];
			      Gold_form_factor[l] +=		Gold_form_factor_account[str_num][l];
			      Mercury_form_factor[l] +=		Mercury_form_factor_account[str_num][l];
			      Thallium_form_factor[l] +=	Thallium_form_factor_account[str_num][l];
			      Lead_form_factor[l] +=		Lead_form_factor_account[str_num][l];
			      Bismuth_form_factor[l] +=		Bismuth_form_factor_account[str_num][l];
			      Polonium_form_factor[l] +=	Polonium_form_factor_account[str_num][l];
			      Astatine_form_factor[l] +=	Astatine_form_factor_account[str_num][l];
			      Radon_form_factor[l] +=		Radon_form_factor_account[str_num][l];
			      Francium_form_factor[l] +=	Francium_form_factor_account[str_num][l];
			      Radium_form_factor[l] +=		Radium_form_factor_account[str_num][l];
			      Actinium_form_factor[l] +=	Actinium_form_factor_account[str_num][l];
			      Thorium_form_factor[l] +=		Thorium_form_factor_account[str_num][l];
			      Protactinium_form_factor[l] +=	Protactinium_form_factor_account[str_num][l];
			      Uranium_form_factor[l] +=		Uranium_form_factor_account[str_num][l];
			      Neptunium_form_factor[l] +=	Neptunium_form_factor_account[str_num][l];
			      Plutonium_form_factor[l] +=	Plutonium_form_factor_account[str_num][l];
			      Americium_form_factor[l] +=	Americium_form_factor_account[str_num][l];
			      Curium_form_factor[l] +=		Curium_form_factor_account[str_num][l];
			      Berkelium_form_factor[l] +=	Berkelium_form_factor_account[str_num][l];
			      Californium_form_factor[l] +=	Californium_form_factor_account[str_num][l];
			      Einsteinium_form_factor[l] +=	Einsteinium_form_factor_account[str_num][l];
			      Fermium_form_factor[l] +=		Fermium_form_factor_account[str_num][l];
			      Mendelevium_form_factor[l] +=	Mendelevium_form_factor_account[str_num][l];
			      Nobelium_form_factor[l] +=	Nobelium_form_factor_account[str_num][l];
			      Lawrencium_form_factor[l] +=	Lawrencium_form_factor_account[str_num][l];
			      Rutherfordium_form_factor[l] +=	Rutherfordium_form_factor_account[str_num][l];
			      Dubnium_form_factor[l] +=		Dubnium_form_factor_account[str_num][l];
			      Seaborgium_form_factor[l] +=	Seaborgium_form_factor_account[str_num][l];
			      Bohrium_form_factor[l] +=		Bohrium_form_factor_account[str_num][l];
			      Hassium_form_factor[l] +=		Hassium_form_factor_account[str_num][l];
			      Meitnerium_form_factor[l] +=	Meitnerium_form_factor_account[str_num][l];
		       //     printf(" %le + i%f\n", creal(f_0[l]),cimag(f_0[l])); 
		    	      }
                         } // pragma ended
		    total_sc = total_sc + 1 ;
//                  clock_gettime(CLOCK_REALTIME, &tht2);
  //                fprintf(stderr,"Checking time %g\n",diff_time(tht1,tht2));
                  //  free(cq);
		     }

        /*************************************************************************************************/
        /*************************************************************************************************/
        /*************************************************************************************************/
           } // for single loop ended for Q-points
            
            int xxx ;
            for (xxx = 0; xxx < qp_rcount ; xxx++)
	      {
           //    fprintf (complex_out, "% d % le \n", xxx, cabs(f_0[xxx])*cabs(f_0[xxx])); 
	         fprintf (complex_out, "% d % e % e % e % e\n", xxx,  cabs(f_0[xxx])*cabs(f_0[xxx]) , creal (f_0[xxx]),cimag (f_0[xxx]),cabs(qrterm_comp[xxx])*cabs(qrterm_comp[xxx]));
                 fprintf (formfactor_out,"% le % le % le % le % le % le % le % le % le % le % le % le % le % le % le % le % le % le % le\
% le % le  % le % le % le % le % le % le % le % le % le % le % le % le % le % le % le % le % le % le % le % le % le % le\
% le % le % le % le % le % le % le % le % le % le % le % le % le % le % le % le % le % le % le % le % le % le % le % le \
%le % le % le % le % le % le % le % le % le % le % le % le % le % le % le % le % le % le % le % le % le % le % le % le % le\
% le % le % le % le % le % le % le % le % le % le % le % le % le % le % le % le % le\n",
			  
			  Hydrogen_form_factor[xxx] 	 /( Hydrogen_number_count * total_sc) ,
			  Helium_form_factor[xxx] 	 /( Helium_number_count * total_sc) , 
			  Lithium_form_factor[xxx] 	 /( Lithium_number_count * total_sc) , 
			  Beryllium_form_factor[xxx] 	 /( Beryllium_number_count * total_sc) , 
			  Boron_form_factor[xxx] 	 /( Boron_number_count * total_sc) , 
			  Carbon_form_factor[xxx]  	 /( Carbon_number_count * total_sc) , 
			  Nitrogen_form_factor[xxx]  	 /( Nitrogen_number_count * total_sc) , 
			  Oxygen_form_factor[xxx] 	 /( Oxygen_number_count * total_sc) , 
			  Fluorine_form_factor[xxx] 	 /( Fluorine_number_count * total_sc) , 
			  Neon_form_factor[xxx] 	 /( Neon_number_count * total_sc) , 
			  Sodium_form_factor[xxx] 	 /( Sodium_number_count * total_sc) , 
			  Magnesium_form_factor[xxx] 	 /( Magnesium_number_count * total_sc) , 
			  Aluminum_form_factor[xxx] 	 /( Aluminum_number_count * total_sc) , 
			  Silicon_form_factor[xxx] 	 /( Silicon_number_count * total_sc) , 
			  Phosphorus_form_factor[xxx] 	 /( Phosphorus_number_count * total_sc) , 
			  Sulfur_form_factor[xxx] 	 /( Sulfur_number_count * total_sc) , 
			  Chlorine_form_factor[xxx] 	 /( Chlorine_number_count * total_sc) , 
			  Argon_form_factor[xxx] 	 /( Argon_number_count * total_sc) , 
			  Potassium_form_factor[xxx] 	 /( Potassium_number_count * total_sc) , 
			  Calcium_form_factor[xxx] 	 /( Calcium_number_count * total_sc) , 
			  Scandium_form_factor[xxx] 	 /( Scandium_number_count * total_sc) , 
			  Titanium_form_factor[xxx] 	 /( Titanium_number_count * total_sc) , 
			  Vanadium_form_factor[xxx] 	 /( Vanadium_number_count * total_sc) , 
			  Chromium_form_factor[xxx] 	 /( Chromium_number_count * total_sc) , 
			  Manganese_form_factor[xxx] 	 /( Manganese_number_count * total_sc) , 
			  Iron_form_factor[xxx] 	 /( Iron_number_count * total_sc) , 
			  Cobalt_form_factor[xxx] 	 /( Cobalt_number_count * total_sc) , 
			  Nickel_form_factor[xxx] 	 /( Nickel_number_count * total_sc) , 
			  Copper_form_factor[xxx] 	 /( Copper_number_count * total_sc) , 
			  Zinc_form_factor[xxx] 	 /( Zinc_number_count * total_sc) , 
			  Gallium_form_factor[xxx] 	 /( Gallium_number_count * total_sc) , 
			  Germanium_form_factor[xxx] 	 /( Germanium_number_count * total_sc) , 
			  Arsenic_form_factor[xxx] 	 /( Arsenic_number_count * total_sc) , 
			  Selenium_form_factor[xxx] 	 /( Selenium_number_count * total_sc) , 
			  Bromine_form_factor[xxx] 	 /( Bromine_number_count * total_sc) , 
			  Krypton_form_factor[xxx] 	 /( Krypton_number_count * total_sc) , 
			  Rubidium_form_factor[xxx] 	 /( Rubidium_number_count * total_sc) , 
			  Strontium_form_factor[xxx] 	 /( Strontium_number_count * total_sc) , 
			  Yttrium_form_factor[xxx] 	 /( Yttrium_number_count * total_sc) , 
			  Zirconium_form_factor[xxx] 	 /( Zirconium_number_count * total_sc) , 
			  Niobium_form_factor[xxx] 	 /( Niobium_number_count * total_sc) , 
			  Molybdenum_form_factor[xxx] 	 /( Molybdenum_number_count * total_sc) , 
			  Technetium_form_factor[xxx] 	 /( Technetium_number_count * total_sc) , 
			  Ruthenium_form_factor[xxx] 	 /( Ruthenium_number_count * total_sc) , 
			  Rhodium_form_factor[xxx] 	 /( Rhodium_number_count * total_sc) , 
			  Palladium_form_factor[xxx] 	 /( Palladium_number_count * total_sc) , 
			  Silver_form_factor[xxx] 	 /( Silver_number_count * total_sc) , 
			  Cadmium_form_factor[xxx] 	 /( Cadmium_number_count * total_sc) , 
			  Indium_form_factor[xxx] 	 /( Indium_number_count * total_sc) , 
			  Tin_form_factor[xxx] 	 /( Tin_number_count * total_sc) , 
			  Antimony_form_factor[xxx] 	 /( Antimony_number_count * total_sc) , 
			  Tellurium_form_factor[xxx] 	 /( Tellurium_number_count * total_sc) , 
			  Iodine_form_factor[xxx] 	  /(Iodine_number_count * total_sc) , 
			  Xenon_form_factor[xxx] 	 /( Xenon_number_count * total_sc) , 
			  Cesium_form_factor[xxx] 	 /( Cesium_number_count * total_sc) , 
			  Barium_form_factor[xxx] 	 /( Barium_number_count * total_sc) , 
			  Lanthanum_form_factor[xxx] 	 /( Lanthanum_number_count * total_sc) , 
			  Cerium_form_factor[xxx] 	 /( Cerium_number_count * total_sc) , 
			  Praseodymium_form_factor[xxx] 	 /( Praseodymium_number_count * total_sc) , 
			  Neodymium_form_factor[xxx] 	 /( Neodymium_number_count * total_sc) , 
			  Promethium_form_factor[xxx] 	 /( Promethium_number_count * total_sc) , 
			  Samarium_form_factor[xxx] 	 /( Samarium_number_count * total_sc) , 
			  Europium_form_factor[xxx] 	 /( Europium_number_count * total_sc) , 
			  Gadolinium_form_factor[xxx] 	 /( Gadolinium_number_count * total_sc) , 
			  Terbium_form_factor[xxx] 	 /( Terbium_number_count * total_sc) , 
			  Dysprosium_form_factor[xxx] 	 /( Dysprosium_number_count * total_sc) , 
			  Holmium_form_factor[xxx] 	 /( Holmium_number_count * total_sc) , 
			  Erbium_form_factor[xxx] 	 /( Erbium_number_count * total_sc) , 
			  Thulium_form_factor[xxx] 	 /( Thulium_number_count * total_sc) , 
			  Ytterbium_form_factor[xxx] 	 /( Ytterbium_number_count * total_sc) , 
			  Lutetium_form_factor[xxx] 	 /( Lutetium_number_count * total_sc) , 
			  Hafnium_form_factor[xxx] 	 /( Hafnium_number_count * total_sc) , 
			  Tantalum_form_factor[xxx] 	 /( Tantalum_number_count * total_sc) , 
			  Tungsten_form_factor[xxx] 	 /( Tungsten_number_count * total_sc) , 
			  Rhenium_form_factor[xxx] 	 /( Rhenium_number_count * total_sc) , 
			  Osmium_form_factor[xxx] 	 /( Osmium_number_count * total_sc) , 
			  Iridium_form_factor[xxx] 	 /( Iridium_number_count * total_sc) , 
			  Platinum_form_factor[xxx] 	 /( Platinum_number_count * total_sc) , 
			  Gold_form_factor[xxx] 	 /( Gold_number_count * total_sc) , 
			  Mercury_form_factor[xxx] 	 /( Mercury_number_count * total_sc) , 
			  Thallium_form_factor[xxx] 	 /( Thallium_number_count * total_sc) , 
			  Lead_form_factor[xxx] 	 /( Lead_number_count * total_sc) , 
			  Bismuth_form_factor[xxx] 	 /( Bismuth_number_count * total_sc) , 
			  Polonium_form_factor[xxx] 	/(  Polonium_number_count * total_sc) , 
			  Astatine_form_factor[xxx] 	 /( Astatine_number_count * total_sc) , 
			  Radon_form_factor[xxx] 	  /(Radon_number_count * total_sc) , 
			  Francium_form_factor[xxx] 	 /( Francium_number_count * total_sc) , 
			  Radium_form_factor[xxx] 	 /( Radium_number_count * total_sc) , 
			  Actinium_form_factor[xxx] 	 /( Actinium_number_count * total_sc) , 
			  Thorium_form_factor[xxx] 	 /( Thorium_number_count * total_sc) , 
			  Protactinium_form_factor[xxx] 	 /( Protactinium_number_count * total_sc) , 
			  Uranium_form_factor[xxx] 	 /( Uranium_number_count * total_sc) , 
			  Neptunium_form_factor[xxx] 	 /( Neptunium_number_count * total_sc) , 
			  Plutonium_form_factor[xxx] 	 /( Plutonium_number_count * total_sc) , 
			  Americium_form_factor[xxx] 	/(  Americium_number_count * total_sc) , 
			  Curium_form_factor[xxx] 	 /( Curium_number_count * total_sc) , 
			  Berkelium_form_factor[xxx] 	 /( Berkelium_number_count * total_sc) , 
			  Californium_form_factor[xxx] 	 /( Californium_number_count * total_sc) , 
			  Einsteinium_form_factor[xxx] 	 /( Einsteinium_number_count * total_sc) , 
			  Fermium_form_factor[xxx] 	 /( Fermium_number_count * total_sc) , 
			  Mendelevium_form_factor[xxx] 	 /( Mendelevium_number_count * total_sc) , 
			  Nobelium_form_factor[xxx] 	 /( Nobelium_number_count * total_sc) , 
			  Lawrencium_form_factor[xxx] 	 /( Lawrencium_number_count * total_sc) , 
			  Rutherfordium_form_factor[xxx] 	 /( Rutherfordium_number_count * total_sc) , 
			  Dubnium_form_factor[xxx] 	 /( Dubnium_number_count * total_sc) , 
			  Seaborgium_form_factor[xxx] 	 /( Seaborgium_number_count * total_sc) , 
			  Bohrium_form_factor[xxx] 	 /( Bohrium_number_count * total_sc) , 
			  Hassium_form_factor[xxx] 	 /( Hassium_number_count * total_sc) , 
			  Meitnerium_form_factor[xxx] )	 /( Meitnerium_number_count * total_sc) ; 
			  

       	      }
       	    
              fclose (complex_out) ;
              fclose (r_info) ;
              fclose (T_info) ;
              fclose (Q_info) ;
              fclose (f0_info) ;
              fclose (fpp_info) ;
              fclose (ele_r_info) ;
              fclose (charge_info) ;
              fclose (atomic_no_info) ;
              free (f_0) ;  
	      free(intensity_account);
              free(qrterm_comp);
              free(qrterm_account);
              

	      free(Hydrogen_form_factor_account);
	      free(Helium_form_factor_account);
	      free(Lithium_form_factor_account);
	      free(Beryllium_form_factor_account);
	      free(Boron_form_factor_account);
	      free(Carbon_form_factor_account);
	      free(Nitrogen_form_factor_account);
	      free(Oxygen_form_factor_account);
	      free(Fluorine_form_factor_account);
	      free(Neon_form_factor_account);
	      free(Sodium_form_factor_account);
	      free(Magnesium_form_factor_account);
	      free(Aluminum_form_factor_account);
	      free(Silicon_form_factor_account);
	      free(Phosphorus_form_factor_account);
	      free(Sulfur_form_factor_account);
	      free(Chlorine_form_factor_account);
	      free(Argon_form_factor_account);
	      free(Potassium_form_factor_account);
	      free(Calcium_form_factor_account);
	      free(Scandium_form_factor_account);
	      free(Titanium_form_factor_account);
	      free(Vanadium_form_factor_account);
	      free(Chromium_form_factor_account);
	      free(Manganese_form_factor_account);
	      free(Iron_form_factor_account);
	      free(Cobalt_form_factor_account);
	      free(Nickel_form_factor_account);
	      free(Copper_form_factor_account);
	      free(Zinc_form_factor_account);
	      free(Gallium_form_factor_account);
	      free(Germanium_form_factor_account);
	      free(Arsenic_form_factor_account);
	      free(Selenium_form_factor_account);
	      free(Bromine_form_factor_account);
	      free(Krypton_form_factor_account);
	      free(Rubidium_form_factor_account);
	      free(Strontium_form_factor_account);
	      free(Yttrium_form_factor_account);
	      free(Zirconium_form_factor_account);
	      free(Niobium_form_factor_account);
	      free(Molybdenum_form_factor_account);
	      free(Technetium_form_factor_account);
	      free(Ruthenium_form_factor_account);
	      free(Rhodium_form_factor_account);
	      free(Palladium_form_factor_account);
	      free(Silver_form_factor_account);
	      free(Cadmium_form_factor_account);
	      free(Indium_form_factor_account);
	      free(Tin_form_factor_account);
	      free(Antimony_form_factor_account);
	      free(Tellurium_form_factor_account);
	      free(Iodine_form_factor_account);
	      free(Xenon_form_factor_account);
	      free(Cesium_form_factor_account);
	      free(Barium_form_factor_account);
	      free(Lanthanum_form_factor_account);
	      free(Cerium_form_factor_account);
	      free(Praseodymium_form_factor_account);
	      free(Neodymium_form_factor_account);
	      free(Promethium_form_factor_account);
	      free(Samarium_form_factor_account);
	      free(Europium_form_factor_account);
	      free(Gadolinium_form_factor_account);
	      free(Terbium_form_factor_account);
	      free(Dysprosium_form_factor_account);
	      free(Holmium_form_factor_account);
	      free(Erbium_form_factor_account);
	      free(Thulium_form_factor_account);
	      free(Ytterbium_form_factor_account);
	      free(Lutetium_form_factor_account);
	      free(Hafnium_form_factor_account);
	      free(Tantalum_form_factor_account);
	      free(Tungsten_form_factor_account);
	      free(Rhenium_form_factor_account);
	      free(Osmium_form_factor_account);
	      free(Iridium_form_factor_account);
	      free(Platinum_form_factor_account);
	      free(Gold_form_factor_account);
	      free(Mercury_form_factor_account);
	      free(Thallium_form_factor_account);
	      free(Lead_form_factor_account);
	      free(Bismuth_form_factor_account);
	      free(Polonium_form_factor_account);
	      free(Astatine_form_factor_account);
	      free(Radon_form_factor_account);
	      free(Francium_form_factor_account);
	      free(Radium_form_factor_account);
	      free(Actinium_form_factor_account);
	      free(Thorium_form_factor_account);
	      free(Protactinium_form_factor_account);
	      free(Uranium_form_factor_account);
	      free(Neptunium_form_factor_account);
	      free(Plutonium_form_factor_account);
	      free(Americium_form_factor_account);
	      free(Curium_form_factor_account);
	      free(Berkelium_form_factor_account);
	      free(Californium_form_factor_account);
	      free(Einsteinium_form_factor_account);
	      free(Fermium_form_factor_account);
	      free(Mendelevium_form_factor_account);
	      free(Nobelium_form_factor_account);
	      free(Lawrencium_form_factor_account);
	      free(Rutherfordium_form_factor_account);
	      free(Dubnium_form_factor_account);
	      free(Seaborgium_form_factor_account);
	      free(Bohrium_form_factor_account);
	      free(Hassium_form_factor_account);
	      free(Meitnerium_form_factor_account);
	      
	                    
	      free(Hydrogen_form_factor);
	      free(Helium_form_factor);
	      free(Lithium_form_factor);
	      free(Beryllium_form_factor);
	      free(Boron_form_factor);
	      free(Carbon_form_factor);
	      free(Nitrogen_form_factor);
	      free(Oxygen_form_factor);
	      free(Fluorine_form_factor);
	      free(Neon_form_factor);
	      free(Sodium_form_factor);
	      free(Magnesium_form_factor);
	      free(Aluminum_form_factor);
	      free(Silicon_form_factor);
	      free(Phosphorus_form_factor);
	      free(Sulfur_form_factor);
	      free(Chlorine_form_factor);
	      free(Argon_form_factor);
	      free(Potassium_form_factor);
	      free(Calcium_form_factor);
	      free(Scandium_form_factor);
	      free(Titanium_form_factor);
	      free(Vanadium_form_factor);
	      free(Chromium_form_factor);
	      free(Manganese_form_factor);
	      free(Iron_form_factor);
	      free(Cobalt_form_factor);
	      free(Nickel_form_factor);
	      free(Copper_form_factor);
	      free(Zinc_form_factor);
	      free(Gallium_form_factor);
	      free(Germanium_form_factor);
	      free(Arsenic_form_factor);
	      free(Selenium_form_factor);
	      free(Bromine_form_factor);
	      free(Krypton_form_factor);
	      free(Rubidium_form_factor);
	      free(Strontium_form_factor);
	      free(Yttrium_form_factor);
	      free(Zirconium_form_factor);
	      free(Niobium_form_factor);
	      free(Molybdenum_form_factor);
	      free(Technetium_form_factor);
	      free(Ruthenium_form_factor);
	      free(Rhodium_form_factor);
	      free(Palladium_form_factor);
	      free(Silver_form_factor);
	      free(Cadmium_form_factor);
	      free(Indium_form_factor);
	      free(Tin_form_factor);
	      free(Antimony_form_factor);
	      free(Tellurium_form_factor);
	      free(Iodine_form_factor);
	      free(Xenon_form_factor);
	      free(Cesium_form_factor);
	      free(Barium_form_factor);
	      free(Lanthanum_form_factor);
	      free(Cerium_form_factor);
	      free(Praseodymium_form_factor);
	      free(Neodymium_form_factor);
	      free(Promethium_form_factor);
	      free(Samarium_form_factor);
	      free(Europium_form_factor);
	      free(Gadolinium_form_factor);
	      free(Terbium_form_factor);
	      free(Dysprosium_form_factor);
	      free(Holmium_form_factor);
	      free(Erbium_form_factor);
	      free(Thulium_form_factor);
	      free(Ytterbium_form_factor);
	      free(Lutetium_form_factor);
	      free(Hafnium_form_factor);
	      free(Tantalum_form_factor);
	      free(Tungsten_form_factor);
	      free(Rhenium_form_factor);
	      free(Osmium_form_factor);
	      free(Iridium_form_factor);
	      free(Platinum_form_factor);
	      free(Gold_form_factor);
	      free(Mercury_form_factor);
	      free(Thallium_form_factor);
	      free(Lead_form_factor);
	      free(Bismuth_form_factor);
	      free(Polonium_form_factor);
	      free(Astatine_form_factor);
	      free(Radon_form_factor);
	      free(Francium_form_factor);
	      free(Radium_form_factor);
	      free(Actinium_form_factor);
	      free(Thorium_form_factor);
	      free(Protactinium_form_factor);
	      free(Uranium_form_factor);
	      free(Neptunium_form_factor);
	      free(Plutonium_form_factor);
	      free(Americium_form_factor);
	      free(Curium_form_factor);
	      free(Berkelium_form_factor);
	      free(Californium_form_factor);
	      free(Einsteinium_form_factor);
	      free(Fermium_form_factor);
	      free(Mendelevium_form_factor);
	      free(Nobelium_form_factor);
	      free(Lawrencium_form_factor);
	      free(Rutherfordium_form_factor);
	      free(Dubnium_form_factor);
	      free(Seaborgium_form_factor);
	      free(Bohrium_form_factor);
	      free(Hassium_form_factor);
	      free(Meitnerium_form_factor);	      
	      
	      
	      
	      fclose (coordinates_out) ;
              fclose (formfactor_out) ;

        


    }	// Timestep loop ended


    
  free(flu_array);
  free(in_fluence);
  free (ccq);
  free(qpt_array);
  free(dummy_array_1);
  free(dummy_array_2);
  free(dummy_array_3);
  free(dummy_array_4);
  free(dummy_array_5);
  free(dummy_array_6);
  free(dummy_array_7);
  free(chk_string);
  return (0);			/* end with proper status   */
   }

}


double yrandom (double xl, double xh)
{

  return (xl + (xh - xl) * ((double) random ()) / 2147483647.0);
}
