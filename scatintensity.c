#include <stdio.h>
#include <stdlib.h>
#include <math.h>  
#include <omp.h>
#include <string.h>
#include <complex.h>
#include <sys/stat.h>
#include "scatintensity.h"



double calc_intensity (int qp_rcount,int i ,int N, int rcount, int lcount, int jj, double *flu_array, int total_sc, int ts,
double T0, double tau, double integrated_signal, double (*f0)[lcount], int *atyp, double *fpp, int  fpp_check, 
double qvect1, double qvect2, double qvect3, double (*cq)[3], double (*CELL)[3], double  scellx, double scelly,
double scellz,int *Z_atoms, int r_ele_variable, int *q_ele, double (*r_ele)[3], double complex *f_0,
double complex *dummy_array_1, double complex *dummy_array_2, double complex *dummy_array_3,
double complex *dummy_array_4, double complex *dummy_array_5,double complex *dummy_array_6,
double complex *dummy_array_7,double complex *qrterm_comp,int *comp_num, int num_threads, int chk_xxx, 
double complex (*intensity_account)[qp_rcount],double complex (*qrterm_account)[qp_rcount] ,int copy_var_count, 
int tpshapef , int tpshapeg, double st_snp, double FinalSnap, double sp_diff, 

			      double (*Hydrogen_form_factor_account)[qp_rcount],
			      double (*Helium_form_factor_account)[qp_rcount],
			      double (*Lithium_form_factor_account)[qp_rcount],
			      double (*Beryllium_form_factor_account)[qp_rcount],
			      double (*Boron_form_factor_account)[qp_rcount],
			      double (*Carbon_form_factor_account)[qp_rcount],
			      double (*Nitrogen_form_factor_account)[qp_rcount],
			      double (*Oxygen_form_factor_account)[qp_rcount],
			      double (*Fluorine_form_factor_account)[qp_rcount],
			      double (*Neon_form_factor_account)[qp_rcount],
			      double (*Sodium_form_factor_account)[qp_rcount],
			      double (*Magnesium_form_factor_account)[qp_rcount],
			      double (*Aluminum_form_factor_account)[qp_rcount],
			      double (*Silicon_form_factor_account)[qp_rcount],
			      double (*Phosphorus_form_factor_account)[qp_rcount],
			      double (*Sulfur_form_factor_account)[qp_rcount],
			      double (*Chlorine_form_factor_account)[qp_rcount],
			      double (*Argon_form_factor_account)[qp_rcount],
			      double (*Potassium_form_factor_account)[qp_rcount],
			      double (*Calcium_form_factor_account)[qp_rcount],
			      double (*Scandium_form_factor_account)[qp_rcount],
			      double (*Titanium_form_factor_account)[qp_rcount],
			      double (*Vanadium_form_factor_account)[qp_rcount],
			      double (*Chromium_form_factor_account)[qp_rcount],
			      double (*Manganese_form_factor_account)[qp_rcount],
			      double (*Iron_form_factor_account)[qp_rcount],
			      double (*Cobalt_form_factor_account)[qp_rcount],
			      double (*Nickel_form_factor_account)[qp_rcount],
			      double (*Copper_form_factor_account)[qp_rcount],
			      double (*Zinc_form_factor_account)[qp_rcount],
			      double (*Gallium_form_factor_account)[qp_rcount],
			      double (*Germanium_form_factor_account)[qp_rcount],
			      double (*Arsenic_form_factor_account)[qp_rcount],
			      double (*Selenium_form_factor_account)[qp_rcount],
			      double (*Bromine_form_factor_account)[qp_rcount],
			      double (*Krypton_form_factor_account)[qp_rcount],
			      double (*Rubidium_form_factor_account)[qp_rcount],
			      double (*Strontium_form_factor_account)[qp_rcount],
			      double (*Yttrium_form_factor_account)[qp_rcount],
			      double (*Zirconium_form_factor_account)[qp_rcount],
			      double (*Niobium_form_factor_account)[qp_rcount],
			      double (*Molybdenum_form_factor_account)[qp_rcount],
			      double (*Technetium_form_factor_account)[qp_rcount],
			      double (*Ruthenium_form_factor_account)[qp_rcount],
			      double (*Rhodium_form_factor_account)[qp_rcount],
			      double (*Palladium_form_factor_account)[qp_rcount],
			      double (*Silver_form_factor_account)[qp_rcount],
			      double (*Cadmium_form_factor_account)[qp_rcount],
			      double (*Indium_form_factor_account)[qp_rcount],
			      double (*Tin_form_factor_account)[qp_rcount],
			      double (*Antimony_form_factor_account)[qp_rcount],
			      double (*Tellurium_form_factor_account)[qp_rcount],
			      double (*Iodine_form_factor_account)[qp_rcount],
			      double (*Xenon_form_factor_account)[qp_rcount],
			      double (*Cesium_form_factor_account)[qp_rcount],
			      double (*Barium_form_factor_account)[qp_rcount],
			      double (*Lanthanum_form_factor_account)[qp_rcount],
			      double (*Cerium_form_factor_account)[qp_rcount],
			      double (*Praseodymium_form_factor_account)[qp_rcount],
			      double (*Neodymium_form_factor_account)[qp_rcount],
			      double (*Promethium_form_factor_account)[qp_rcount],
			      double (*Samarium_form_factor_account)[qp_rcount],
			      double (*Europium_form_factor_account)[qp_rcount],
			      double (*Gadolinium_form_factor_account)[qp_rcount],
			      double (*Terbium_form_factor_account)[qp_rcount],
			      double (*Dysprosium_form_factor_account)[qp_rcount],
			      double (*Holmium_form_factor_account)[qp_rcount],
			      double (*Erbium_form_factor_account)[qp_rcount],
			      double (*Thulium_form_factor_account)[qp_rcount],
			      double (*Ytterbium_form_factor_account)[qp_rcount],
			      double (*Lutetium_form_factor_account)[qp_rcount],
			      double (*Hafnium_form_factor_account)[qp_rcount],
			      double (*Tantalum_form_factor_account)[qp_rcount],
			      double (*Tungsten_form_factor_account)[qp_rcount],
			      double (*Rhenium_form_factor_account)[qp_rcount],
			      double (*Osmium_form_factor_account)[qp_rcount],
			      double (*Iridium_form_factor_account)[qp_rcount],
			      double (*Platinum_form_factor_account)[qp_rcount],
			      double (*Gold_form_factor_account)[qp_rcount],
			      double (*Mercury_form_factor_account)[qp_rcount],
			      double (*Thallium_form_factor_account)[qp_rcount],
			      double (*Lead_form_factor_account)[qp_rcount],
			      double (*Bismuth_form_factor_account)[qp_rcount],
			      double (*Polonium_form_factor_account)[qp_rcount],
			      double (*Astatine_form_factor_account)[qp_rcount],
			      double (*Radon_form_factor_account)[qp_rcount],
			      double (*Francium_form_factor_account)[qp_rcount],
			      double (*Radium_form_factor_account)[qp_rcount],
			      double (*Actinium_form_factor_account)[qp_rcount],
			      double (*Thorium_form_factor_account)[qp_rcount],
			      double (*Protactinium_form_factor_account)[qp_rcount],
			      double (*Uranium_form_factor_account)[qp_rcount],
			      double (*Neptunium_form_factor_account)[qp_rcount],
			      double (*Plutonium_form_factor_account)[qp_rcount],
			      double (*Americium_form_factor_account)[qp_rcount],
			      double (*Curium_form_factor_account)[qp_rcount],
			      double (*Berkelium_form_factor_account)[qp_rcount],
			      double (*Californium_form_factor_account)[qp_rcount],
			      double (*Einsteinium_form_factor_account)[qp_rcount],
			      double (*Fermium_form_factor_account)[qp_rcount],
			      double (*Mendelevium_form_factor_account)[qp_rcount],
			      double (*Nobelium_form_factor_account)[qp_rcount],
			      double (*Lawrencium_form_factor_account)[qp_rcount],
			      double (*Rutherfordium_form_factor_account)[qp_rcount],
			      double (*Dubnium_form_factor_account)[qp_rcount],
			      double (*Seaborgium_form_factor_account)[qp_rcount],
			      double (*Bohrium_form_factor_account)[qp_rcount],
			      double (*Hassium_form_factor_account)[qp_rcount],
			      double (*Meitnerium_form_factor_account)[qp_rcount],
			      
			      
			      double *Hydrogen_form_factor,
			      double *Helium_form_factor,
			      double *Lithium_form_factor,
			      double *Beryllium_form_factor,
			      double *Boron_form_factor,
			      double *Carbon_form_factor,
			      double *Nitrogen_form_factor,
			      double *Oxygen_form_factor,
			      double *Fluorine_form_factor,
			      double *Neon_form_factor,
			      double *Sodium_form_factor,
			      double *Magnesium_form_factor,
			      double *Aluminum_form_factor,
			      double *Silicon_form_factor,
			      double *Phosphorus_form_factor,
			      double *Sulfur_form_factor,
			      double *Chlorine_form_factor,
			      double *Argon_form_factor,
			      double *Potassium_form_factor,
			      double *Calcium_form_factor,
			      double *Scandium_form_factor,
			      double *Titanium_form_factor,
			      double *Vanadium_form_factor,
			      double *Chromium_form_factor,
			      double *Manganese_form_factor,
			      double *Iron_form_factor,
			      double *Cobalt_form_factor,
			      double *Nickel_form_factor,
			      double *Copper_form_factor,
			      double *Zinc_form_factor,
			      double *Gallium_form_factor,
			      double *Germanium_form_factor,
			      double *Arsenic_form_factor,
			      double *Selenium_form_factor,
			      double *Bromine_form_factor,
			      double *Krypton_form_factor,
			      double *Rubidium_form_factor,
			      double *Strontium_form_factor,
			      double *Yttrium_form_factor,
			      double *Zirconium_form_factor,
			      double *Niobium_form_factor,
			      double *Molybdenum_form_factor,
			      double *Technetium_form_factor,
			      double *Ruthenium_form_factor,
			      double *Rhodium_form_factor,
			      double *Palladium_form_factor,
			      double *Silver_form_factor,
			      double *Cadmium_form_factor,
			      double *Indium_form_factor,
			      double *Tin_form_factor,
			      double *Antimony_form_factor,
			      double *Tellurium_form_factor,
			      double *Iodine_form_factor,
			      double *Xenon_form_factor,
			      double *Cesium_form_factor,
			      double *Barium_form_factor,
			      double *Lanthanum_form_factor,
			      double *Cerium_form_factor,
			      double *Praseodymium_form_factor,
			      double *Neodymium_form_factor,
			      double *Promethium_form_factor,
			      double *Samarium_form_factor,
			      double *Europium_form_factor,
			      double *Gadolinium_form_factor,
			      double *Terbium_form_factor,
			      double *Dysprosium_form_factor,
			      double *Holmium_form_factor,
			      double *Erbium_form_factor,
			      double *Thulium_form_factor,
			      double *Ytterbium_form_factor,
			      double *Lutetium_form_factor,
			      double *Hafnium_form_factor,
			      double *Tantalum_form_factor,
			      double *Tungsten_form_factor,
			      double *Rhenium_form_factor,
			      double *Osmium_form_factor,
			      double *Iridium_form_factor,
			      double *Platinum_form_factor,
			      double *Gold_form_factor,
			      double *Mercury_form_factor,
			      double *Thallium_form_factor,
			      double *Lead_form_factor,
			      double *Bismuth_form_factor,
			      double *Polonium_form_factor,
			      double *Astatine_form_factor,
			      double *Radon_form_factor,
			      double *Francium_form_factor,
			      double *Radium_form_factor,
			      double *Actinium_form_factor,
			      double *Thorium_form_factor,
			      double *Protactinium_form_factor,
			      double *Uranium_form_factor,
			      double *Neptunium_form_factor,
			      double *Plutonium_form_factor,
			      double *Americium_form_factor,
			      double *Curium_form_factor,
			      double *Berkelium_form_factor,
			      double *Californium_form_factor,
			      double *Einsteinium_form_factor,
			      double *Fermium_form_factor,
			      double *Mendelevium_form_factor,
			      double *Nobelium_form_factor,
			      double *Lawrencium_form_factor,
			      double *Rutherfordium_form_factor,
			      double *Dubnium_form_factor,
			      double *Seaborgium_form_factor,
			      double *Bohrium_form_factor,
			      double *Hassium_form_factor,
			      double *Meitnerium_form_factor)
{


 int ii , my_id=0 ;
 double fluence_part=0 ,ff_chk_part=0 ;
 double complex f0_part=0 , exp_part=0 , exp_intensity_part=0  ;
 
 double dummy_T0_2 = T0 / 1000 ;
 double dummy_st_snp_2 = st_snp/1000 ;
 double dummy_FinalSnap_2 = FinalSnap / 1000 ;
 double dummy_sp_diff_2 = sp_diff / 1000 ;
 double dummy_tau_2 = tau / 1000 ;
 double dummy_ts = (ts / 1000) - (st_snp/1000) ;


 fluence_part = ( sqrt(flu_array[total_sc]) * sqrt( exp(-((dummy_ts-dummy_T0_2) * 
 (dummy_ts-dummy_T0_2) )/((dummy_tau_2*dummy_tau_2)/(4 * log(2))))/integrated_signal) * tpshapeg) + ( tpshapef * sqrt(flu_array[total_sc]));

 //printf("% le\n",fluence_part);
// fluence_part = ( sqrt(flu_array[total_sc]) * sqrt( exp(-((ts-T0)*(ts-T0))/(tau*tau))/integrated_signal) * tpshapeg) +
  //                                                               ( tpshapef * sqrt(flu_array[total_sc])) ;


    #pragma omp parallel for private(f0_part,ff_chk_part,exp_part,exp_intensity_part,my_id,ii)
      for ( ii = 0; ii < N; ii++ )
        {
	 my_id = omp_get_thread_num() ;
         f0_part = f0[atyp[ii]][jj] + (fpp[atyp[ii]] * I * fpp_check)  ;
         ff_chk_part = f0[atyp[ii]][jj] ;
         exp_part = cexp((qvect1 * (cq[ii][0] + CELL[i][0] * scellx)
                         + qvect2 * (cq[ii][1] + CELL[i][1] * scelly)
                         + qvect3 * (cq[ii][2] + CELL[i][2] * scellz)) * I) ;
                      //   + qvect3 * (cq[ii][2] + CELL[i][2] * scellz)) * I) * (Z_atoms[ii] / 7 )  ;
         exp_intensity_part = cexp((qvect1 * (cq[ii][0] )+
                                     qvect2 * (cq[ii][1] )+
                                     qvect3 * (cq[ii][2] )) * I) ;
                          //          qvect3 * (cq[ii][2] )) * I) * (Z_atoms[ii] / 7);
         dummy_array_1[my_id] += fluence_part * exp_part * f0_part ;
         dummy_array_2[my_id] += fluence_part * exp_intensity_part * f0_part ;
         dummy_array_5[my_id] += fluence_part * exp_part;
         dummy_array_6[my_id] += fluence_part * exp_intensity_part;
         dummy_array_7[my_id] *= ff_chk_part ;
         //dummy_array_3[my_id] += fluence_part * exp_part ;
        // dummy_array_4[my_id] += fluence_part * exp_intensity_part ;
	
        }
        

      #pragma omp parallel for private(exp_part,exp_intensity_part,my_id,ii)
      for ( ii = 0; ii < r_ele_variable; ii++ )
        {
         my_id = omp_get_thread_num() ;

         exp_part = -q_ele[ii] * cexp ((qvect1 * (r_ele[ii][0] + CELL[i][0] * scellx)
                         + qvect2 * (r_ele[ii][1] + CELL[i][1] * scelly)
                         + qvect3 * (r_ele[ii][2] + CELL[i][2] * scellz)) * I) ;
         exp_intensity_part = -q_ele[ii] * cexp ((qvect1 * (r_ele[ii][0] )+
                                     qvect2 * (r_ele[ii][1] )+
                                     qvect3 * (r_ele[ii][2] )) * I) ;
  
         dummy_array_3[my_id] += fluence_part * exp_part ;
         dummy_array_4[my_id] += fluence_part * exp_intensity_part ;
        }


        for ( my_id=0 ; my_id < num_threads ; my_id++ ) 
         {
          f_0[*comp_num] += (dummy_array_1[my_id] + dummy_array_3[my_id]) ;
       //   f_0[*comp_num] += (dummy_array_1[my_id]) ;
       
       
         intensity_account[copy_var_count][(*comp_num)] +=  (dummy_array_2[my_id] + 
                                                             dummy_array_4[my_id]) ;
          
       //   intensity_account[copy_var_count][(*comp_num)] +=  dummy_array_2[my_id] ; 
                                                              
          qrterm_comp[*comp_num] +=  (dummy_array_5[my_id] +
                                     dummy_array_3[my_id])  ;

      //    qrterm_comp[*comp_num] +=  dummy_array_5[my_id] ;
                                      
          qrterm_account[copy_var_count][(*comp_num)] +=  (dummy_array_6[my_id] +
                                                           dummy_array_4[my_id]) ;

        //  qrterm_account[copy_var_count][(*comp_num)] +=  dummy_array_6[my_id] ;
                                                            

         }
         
double Hydrogen_count = 0 ,
	Helium_count = 0 ,
	Lithium_count = 0 ,
	Beryllium_count = 0 , 
	Boron_count = 0 , 
	Carbon_count = 0 ,
	Nitrogen_count = 0 ,
	Oxygen_count = 0 ,
	Fluorine_count = 0 ,
	Neon_count = 0 ,
	Sodium_count = 0 ,
	Magnesium_count = 0 ,
	Aluminum_count = 0 ,
	Silicon_count = 0 ,
	Phosphorus_count = 0 ,
	Sulfur_count = 0 ,
	Chlorine_count = 0 ,
	Argon_count = 0 ,
	Potassium_count = 0 ,
	Calcium_count = 0 ,
	Scandium_count = 0 ,
	Titanium_count = 0 ,
	Vanadium_count = 0 ,
	Chromium_count = 0 ,
	Manganese_count = 0 ,
	Iron_count = 0 ,
	Cobalt_count = 0 ,
	Nickel_count = 0 ,
	Copper_count = 0 ,
	Zinc_count = 0 ,
	Gallium_count = 0 ,
	Germanium_count = 0 ,
	Arsenic_count = 0 ,
	Selenium_count = 0 ,
	Bromine_count = 0 ,
	Krypton_count = 0 ,
	Rubidium_count = 0 ,
	Strontium_count = 0 ,
	Yttrium_count = 0 ,
	Zirconium_count = 0 ,
	Niobium_count = 0 ,
	Molybdenum_count = 0 ,
	Technetium_count = 0 ,
	Ruthenium_count = 0 ,
	Rhodium_count = 0 ,
	Palladium_count = 0 ,
	Silver_count = 0 ,
	Cadmium_count = 0 ,
	Indium_count = 0 ,
	Tin_count = 0 ,
	Antimony_count = 0 ,
	Tellurium_count = 0 ,
	Iodine_count = 0 ,
	Xenon_count = 0 ,
	Cesium_count = 0 ,
	Barium_count = 0 ,
	Lanthanum_count = 0 ,
	Cerium_count = 0 ,
	Praseodymium_count = 0 ,
	Neodymium_count = 0 ,
	Promethium_count = 0 ,
	Samarium_count = 0 ,
	Europium_count = 0 ,
	Gadolinium_count = 0 ,
	Terbium_count = 0 ,
	Dysprosium_count = 0 ,
	Holmium_count = 0 ,
	Erbium_count = 0 ,
	Thulium_count = 0 ,
	Ytterbium_count = 0 ,
	Lutetium_count = 0 ,
	Hafnium_count = 0 ,
	Tantalum_count = 0 ,
	Tungsten_count = 0 ,
	Rhenium_count = 0 ,
	Osmium_count = 0 ,
	Iridium_count = 0 ,
	Platinum_count = 0 ,
	Gold_count = 0 ,
	Mercury_count = 0 ,
	Thallium_count = 0 ,
	Lead_count = 0 ,
	Bismuth_count = 0 ,
	Polonium_count = 0 ,
	Astatine_count = 0 ,
	Radon_count = 0 ,
	Francium_count = 0 ,
	Radium_count = 0 ,
	Actinium_count = 0 ,
	Thorium_count = 0 ,
	Protactinium_count = 0 ,
	Uranium_count = 0 ,
	Neptunium_count = 0 ,
	Plutonium_count = 0 ,
	Americium_count = 0 ,
	Curium_count = 0 ,
	Berkelium_count = 0 ,
	Californium_count = 0 ,
	Einsteinium_count = 0 ,
	Fermium_count = 0 ,
	Mendelevium_count = 0 ,
	Nobelium_count = 0 ,
	Lawrencium_count = 0 ,
	Rutherfordium_count = 0 ,
	Dubnium_count = 0 ,
	Seaborgium_count = 0 ,
	Bohrium_count = 0 ,
	Hassium_count = 0 ,
	Meitnerium_count = 0 ;
	
	
             for (ii = 0; ii < N ; ii++)
      {
	      if (1 == Z_atoms[ii])
	      {
	      Hydrogen_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Hydrogen_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }

	      if (2 == Z_atoms[ii])
	      {
	      Helium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Helium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (3 == Z_atoms[ii])
	      {
	      Lithium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 
	      Lithium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (4 == Z_atoms[ii])
	      {
	      Beryllium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Beryllium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (5 == Z_atoms[ii])
	      {
	      Boron_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Boron_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (6 == Z_atoms[ii])
	      {
	      Carbon_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Carbon_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (7 == Z_atoms[ii])
	      {
	      Nitrogen_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Nitrogen_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (8 == Z_atoms[ii])
	      {
	      Oxygen_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Oxygen_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (9 == Z_atoms[ii])
	      {
	      Fluorine_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Fluorine_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (10 == Z_atoms[ii])
	      {
	      Neon_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Neon_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (11 == Z_atoms[ii])
	      {
	      Sodium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Sodium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (12 == Z_atoms[ii])
	      {
	      Magnesium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Magnesium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (13 == Z_atoms[ii])
	      {
	      Aluminum_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Aluminum_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (14 == Z_atoms[ii])
	      {
	      Silicon_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Silicon_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (15 == Z_atoms[ii])
	      {
	      Phosphorus_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Phosphorus_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (16 == Z_atoms[ii])
	      {
	      Sulfur_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Sulfur_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (17 == Z_atoms[ii])
	      {
	      Chlorine_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Chlorine_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (18 == Z_atoms[ii])
	      {
	      Argon_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Argon_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (19 == Z_atoms[ii])
	      {
	      Potassium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Potassium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (20 == Z_atoms[ii])
	      {
	      Calcium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Calcium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (21 == Z_atoms[ii])
	      {
	      Scandium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Scandium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (22 == Z_atoms[ii])
	      {
	      Titanium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Titanium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (23 == Z_atoms[ii])
	      {
	      Vanadium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Vanadium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (24 == Z_atoms[ii])
	      {
	      Chromium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Chromium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (25 == Z_atoms[ii])
	      {
	      Manganese_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Manganese_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (26 == Z_atoms[ii])
	      {
	      Iron_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Iron_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (27 == Z_atoms[ii])
	      {
	      Cobalt_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Cobalt_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (28 == Z_atoms[ii])
	      {
	      Nickel_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Nickel_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (29 == Z_atoms[ii])
	      {
	      Copper_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Copper_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (30 == Z_atoms[ii])
	      {
	      Zinc_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Zinc_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (31 == Z_atoms[ii])
	      {
	      Gallium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Gallium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (32 == Z_atoms[ii])
	      {
	      Germanium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Germanium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (33 == Z_atoms[ii])
	      {
	      Arsenic_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Arsenic_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (34 == Z_atoms[ii])
	      {
	      Selenium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Selenium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (35 == Z_atoms[ii])
	      {
	      Bromine_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Bromine_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (36 == Z_atoms[ii])
	      {
	      Krypton_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Krypton_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (37 == Z_atoms[ii])
	      {
	      Rubidium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Rubidium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (38 == Z_atoms[ii])
	      {
	      Strontium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Strontium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (39 == Z_atoms[ii])
	      {
	      Yttrium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Yttrium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (40 == Z_atoms[ii])
	      {
	      Zirconium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Zirconium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (41 == Z_atoms[ii])
	      {
	      Niobium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Niobium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (42 == Z_atoms[ii])
	      {
	      Molybdenum_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Molybdenum_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (43 == Z_atoms[ii])
	      {
	      Technetium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Technetium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (44 == Z_atoms[ii])
	      {
	      Ruthenium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Ruthenium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (45 == Z_atoms[ii])
	      {
	      Rhodium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Rhodium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (46 == Z_atoms[ii])
	      {
	      Palladium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Palladium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (47 == Z_atoms[ii])
	      {
	      Silver_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Silver_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (48 == Z_atoms[ii])
	      {
	      Cadmium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Cadmium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (49 == Z_atoms[ii])
	      {
	      Indium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Indium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (50 == Z_atoms[ii])
	      {
	      Tin_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Tin_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (51 == Z_atoms[ii])
	      {
	      Antimony_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Antimony_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (52 == Z_atoms[ii])
	      {
	      Tellurium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Tellurium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (53 == Z_atoms[ii])
	      {
	      Iodine_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Iodine_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (54 == Z_atoms[ii])
	      {
	      Xenon_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Xenon_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (55 == Z_atoms[ii])
	      {
	      Cesium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Cesium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (56 == Z_atoms[ii])
	      {
	      Barium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Barium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (57 == Z_atoms[ii])
	      {
	      Lanthanum_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Lanthanum_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (58 == Z_atoms[ii])
	      {
	      Cerium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Cerium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (59 == Z_atoms[ii])
	      {
	      Praseodymium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Praseodymium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (60 == Z_atoms[ii])
	      {
	      Neodymium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Neodymium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (61 == Z_atoms[ii])
	      {
	      Promethium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Promethium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (62 == Z_atoms[ii])
	      {
	      Samarium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Samarium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (63 == Z_atoms[ii])
	      {
	      Europium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Europium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (64 == Z_atoms[ii])
	      {
	      Gadolinium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Gadolinium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (65 == Z_atoms[ii])
	      {
	      Terbium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Terbium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (66 == Z_atoms[ii])
	      {
	      Dysprosium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Dysprosium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (67 == Z_atoms[ii])
	      {
	      Holmium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Holmium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (68 == Z_atoms[ii])
	      {
	      Erbium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Erbium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (69 == Z_atoms[ii])
	      {
	      Thulium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Thulium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (70 == Z_atoms[ii])
	      {
	      Ytterbium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Ytterbium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (71 == Z_atoms[ii])
	      {
	      Lutetium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Lutetium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (72 == Z_atoms[ii])
	      {
	      Hafnium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Hafnium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (73 == Z_atoms[ii])
	      {
	      Tantalum_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Tantalum_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (74 == Z_atoms[ii])
	      {
	      Tungsten_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Tungsten_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (75 == Z_atoms[ii])
	      {
	      Rhenium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Rhenium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (76 == Z_atoms[ii])
	      {
	      Osmium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Osmium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (77 == Z_atoms[ii])
	      {
	      Iridium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Iridium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (78 == Z_atoms[ii])
	      {
	      Platinum_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Platinum_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (79 == Z_atoms[ii])
	      {
	      Gold_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Gold_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (80 == Z_atoms[ii])
	      {
	      Mercury_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Mercury_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (81 == Z_atoms[ii])
	      {
	      Thallium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Thallium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (82 == Z_atoms[ii])
	      {
	      Lead_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Lead_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (83 == Z_atoms[ii])
	      {
	      Bismuth_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Bismuth_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (84 == Z_atoms[ii])
	      {
	      Polonium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Polonium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (85 == Z_atoms[ii])
	      {
	      Astatine_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Astatine_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (86 == Z_atoms[ii])
	      {
	      Radon_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Radon_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (87 == Z_atoms[ii])
	      {
	      Francium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Francium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (88 == Z_atoms[ii])
	      {
	      Radium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Radium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (89 == Z_atoms[ii])
	      {
	      Actinium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Actinium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (90 == Z_atoms[ii])
	      {
	      Thorium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Thorium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (91 == Z_atoms[ii])
	      {
	      Protactinium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Protactinium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (92 == Z_atoms[ii])
	      {
	      Uranium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Uranium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (93 == Z_atoms[ii])
	      {
	      Neptunium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Neptunium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (94 == Z_atoms[ii])
	      {
	      Plutonium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Plutonium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (95 == Z_atoms[ii])
	      {
	      Americium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Americium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (96 == Z_atoms[ii])
	      {
	      Curium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Curium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (97 == Z_atoms[ii])
	      {
	      Berkelium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Berkelium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (98 == Z_atoms[ii])
	      {
	      Californium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Californium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (99 == Z_atoms[ii])
	      {
	      Einsteinium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Einsteinium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (100 == Z_atoms[ii])
	      {
	      Fermium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Fermium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (101 == Z_atoms[ii])
	      {
	      Mendelevium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Mendelevium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (102 == Z_atoms[ii])
	      {
	      Nobelium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Nobelium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (103 == Z_atoms[ii])
	      {
	      Lawrencium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Lawrencium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (104 == Z_atoms[ii])
	      {
	      Rutherfordium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Rutherfordium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (105 == Z_atoms[ii])
	      {
	      Dubnium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Dubnium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (106 == Z_atoms[ii])
	      {
	      Seaborgium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Seaborgium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (107 == Z_atoms[ii])
	      {
	      Bohrium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Bohrium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (108 == Z_atoms[ii])
	      {
	      Hassium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Hassium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }


	      if (109 == Z_atoms[ii])
	      {
	      Meitnerium_form_factor[*comp_num] += f0[atyp[ii]][jj] ; 	
	      Meitnerium_form_factor_account[copy_var_count][(*comp_num)] += f0[atyp[ii]][jj] ;
	      }

         
      }
         
         
          (*comp_num)++ ;
 

}          
