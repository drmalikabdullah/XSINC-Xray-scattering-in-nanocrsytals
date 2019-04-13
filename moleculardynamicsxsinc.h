extern int md_integrator_verlet( int N , double (*r)[3] , double (*v)[3] , double *q , double DT , int N_steps ,
double a ,double  b ,double  c ,double  r0 , double  alpha , double  rcut , double  kcut , int use_brute_force , 
				 int use_minimg , char hard_wall[3]);