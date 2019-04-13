


extern int coordinates_atoms( int N, double (*cq)[3] , char *ee) ;
extern int species_atoms( int N , int *atyp , char *ee) ;
extern int f0_atoms(int N , int rcount , int lcount ,double (*f0)[lcount] , char *ee);
extern int Q_atoms(int N , int lcount ,double *Q , char *ee);
extern int fpp_atoms(int rcount , double *fpp ,char *ee , int fpp_check) ;
extern int electron_coordinates(int N , int r_ele_variable ,  double (*r_ele)[3], char *ee) ;
extern int charge_on_electrons(int N , int r_ele_variable , int *q_ele , char *ee) ;
extern int atomic_num_atoms(int N , int *Z_atoms , char *ee) ;
extern int f0_mod_atoms(int N ,int rcount , int lcount ,double (*f0)[lcount] , char *ee_f0mod);
extern int species_atoms_mod(int N , int *atyp , char *ee_atypmod);
